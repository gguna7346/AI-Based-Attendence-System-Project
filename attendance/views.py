import base64
import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.base import ContentFile
from django.db.models import Count

from .models import AttendanceRecord, FaceImage, Student
from .ml.liveness import check_liveness
from .ml.recognition import (
    LabeledEmbedding,
    extract_face_embedding,
    extract_face_embedding_from_bgr,
    json_to_vector,
    recognize_face,
)


def _find_student_from_message(message: str) -> Optional[Student]:
    message_lower = message.lower()
    roll_match = re.search(r"roll\s*(?:no|number)?\s*[:\-]?\s*([a-zA-Z0-9_-]+)", message_lower)
    if roll_match:
        roll = roll_match.group(1)
        return Student.objects.filter(roll_no__iexact=roll).first()

    for student in Student.objects.all():
        if student.roll_no.lower() in message_lower or student.name.lower() in message_lower:
            return student
    return None


def _student_attendance_summary(student: Student) -> str:
    total_class_days = (
        AttendanceRecord.objects.values_list("date", flat=True).distinct().count()
    )
    present_days = (
        AttendanceRecord.objects.filter(
            student=student, status=AttendanceRecord.STATUS_PRESENT
        )
        .values_list("date", flat=True)
        .distinct()
        .count()
    )
    rate = (present_days / total_class_days * 100.0) if total_class_days > 0 else 0.0
    latest_record = AttendanceRecord.objects.filter(student=student).order_by("-date", "-timestamp").first()
    last_seen = latest_record.date.isoformat() if latest_record else "No attendance record yet"
    return (
        f"Attendance for {student.name} ({student.roll_no}): "
        f"{present_days}/{total_class_days} days present ({rate:.1f}%). "
        f"Last attendance date: {last_seen}."
    )


def _absent_today_summary() -> str:
    today = date.today()
    present_ids = (
        AttendanceRecord.objects.filter(date=today, status=AttendanceRecord.STATUS_PRESENT)
        .values_list("student_id", flat=True)
        .distinct()
    )
    absent_students = Student.objects.exclude(id__in=present_ids).order_by("roll_no")
    if not absent_students.exists():
        return f"Great news: nobody is absent today ({today.isoformat()})."

    names = [f"{s.name} ({s.roll_no})" for s in absent_students[:15]]
    remainder = absent_students.count() - len(names)
    suffix = f" and {remainder} more." if remainder > 0 else "."
    return (
        f"Absent today ({today.isoformat()}): {', '.join(names)}{suffix}"
    )


def _present_today_summary() -> str:
    today = date.today()
    present_count = (
        AttendanceRecord.objects.filter(date=today, status=AttendanceRecord.STATUS_PRESENT)
        .values_list("student_id", flat=True)
        .distinct()
        .count()
    )
    total_students = Student.objects.count()
    return (
        f"Present today ({today.isoformat()}): {present_count} out of {total_students} students."
    )


def _build_local_chatbot_response(message: str) -> str:
    normalized = message.lower().strip()
    if not normalized:
        return "Please type a question, for example: 'Who is absent today?' or 'Show attendance for roll no 101'."

    if "absent" in normalized and "today" in normalized:
        return _absent_today_summary()

    if ("present" in normalized and "today" in normalized) or "how many attended today" in normalized:
        return _present_today_summary()

    if "show my attendance" in normalized or "my attendance" in normalized or "attendance of" in normalized or "attendance for" in normalized:
        student = _find_student_from_message(message)
        if student is None:
            return (
                "I can show attendance, but I need a student reference. "
                "Try: 'Show attendance for roll no 101' or 'Show attendance for John'."
            )
        return _student_attendance_summary(student)

    if "insight" in normalized or "risk" in normalized or "prediction" in normalized:
        insights = _compute_smart_insights()
        return (
            f"Smart insight summary: average attendance is {insights['avg_attendance_rate']}%, "
            f"predicted next-week attendance is {insights['predicted_overall_rate']}%, "
            f"and high-risk students are {insights['high_risk_count']}."
        )

    return (
        "I can help with attendance queries. Try:\n"
        "- Who is absent today?\n"
        "- Show attendance for roll no 101\n"
        "- How many students are present today?\n"
        "- Show prediction insights"
    )


def _try_openai_response(message: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        context = (
            f"Today: {date.today().isoformat()}. "
            f"Total students: {Student.objects.count()}. "
            f"{_present_today_summary()} "
            f"{_absent_today_summary()}"
        )
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You are an attendance assistant for a college. Answer briefly and clearly.",
                },
                {
                    "role": "system",
                    "content": f"Context from local database: {context}",
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
        )
        text = getattr(response, "output_text", "").strip()
        return text or None
    except Exception:
        return None


def _risk_label(rate: float) -> str:
    if rate < 60:
        return "High Risk"
    if rate < 75:
        return "Medium Risk"
    return "Low Risk"


def _compute_smart_insights() -> Dict[str, Any]:
    """
    Build lightweight prediction insights from attendance history.
    """
    students = list(Student.objects.all().order_by("roll_no"))
    class_days_qs = (
        AttendanceRecord.objects.values_list("date", flat=True).distinct().order_by("date")
    )
    class_days = list(class_days_qs)
    total_class_days = len(class_days)

    insights_rows: List[Dict[str, Any]] = []
    predicted_rates: List[float] = []

    for student in students:
        present_days_qs = (
            AttendanceRecord.objects.filter(
                student=student, status=AttendanceRecord.STATUS_PRESENT
            )
            .values_list("date", flat=True)
            .distinct()
        )
        present_days_set = set(present_days_qs)
        present_days = len(present_days_set)

        if total_class_days > 0:
            attendance_rate = (present_days / total_class_days) * 100.0
        else:
            attendance_rate = 0.0

        # Trend: compare presence rate in last 7 class days vs previous 7 class days
        last_7_days = class_days[-7:]
        prev_7_days = class_days[-14:-7]
        last_7_present = sum(1 for d in last_7_days if d in present_days_set)
        prev_7_present = sum(1 for d in prev_7_days if d in present_days_set)

        last_7_rate = (last_7_present / len(last_7_days) * 100.0) if last_7_days else attendance_rate
        prev_7_rate = (prev_7_present / len(prev_7_days) * 100.0) if prev_7_days else attendance_rate
        trend = last_7_rate - prev_7_rate

        predicted_next_week_rate = max(0.0, min(100.0, attendance_rate + (trend * 0.5)))
        predicted_rates.append(predicted_next_week_rate)

        risk = _risk_label(attendance_rate)
        if trend < -15:
            # Escalate one level for sharply declining trend
            if risk == "Low Risk":
                risk = "Medium Risk"
            elif risk == "Medium Risk":
                risk = "High Risk"

        insights_rows.append(
            {
                "student": student,
                "present_days": present_days,
                "attendance_rate": round(attendance_rate, 1),
                "last_7_rate": round(last_7_rate, 1),
                "trend": round(trend, 1),
                "predicted_next_week_rate": round(predicted_next_week_rate, 1),
                "risk": risk,
            }
        )

    insights_rows.sort(key=lambda row: row["attendance_rate"])
    high_risk_count = sum(1 for row in insights_rows if row["risk"] == "High Risk")
    avg_attendance_rate = (
        round(sum(row["attendance_rate"] for row in insights_rows) / len(insights_rows), 1)
        if insights_rows
        else 0.0
    )
    predicted_overall_rate = (
        round(sum(predicted_rates) / len(predicted_rates), 1) if predicted_rates else 0.0
    )

    return {
        "total_class_days": total_class_days,
        "avg_attendance_rate": avg_attendance_rate,
        "predicted_overall_rate": predicted_overall_rate,
        "high_risk_count": high_risk_count,
        "insights_rows": insights_rows,
    }


def dashboard(request: HttpRequest) -> HttpResponse:
    total_students = Student.objects.count()
    today = date.today()
    today_present = AttendanceRecord.objects.filter(date=today, status=AttendanceRecord.STATUS_PRESENT).count()
    total_records_today = AttendanceRecord.objects.filter(date=today).count()
    smart_insights = _compute_smart_insights()

    context: Dict[str, Any] = {
        "total_students": total_students,
        "today_present": today_present,
        "today_total_records": total_records_today,
        "today_date": today,
        "total_class_days": smart_insights["total_class_days"],
        "avg_attendance_rate": smart_insights["avg_attendance_rate"],
        "predicted_overall_rate": smart_insights["predicted_overall_rate"],
        "high_risk_count": smart_insights["high_risk_count"],
        "insights_rows": smart_insights["insights_rows"],
    }
    return render(request, "attendance/dashboard.html", context)


def _decode_data_url(image_data: str) -> Tuple[bytes, str]:
    """Decode a base64 data URL into raw bytes and a file extension."""
    if not image_data.startswith("data:image"):
        raise ValueError("Invalid image data")

    header, base64_data = image_data.split(",", 1)
    file_ext = "png" if "png" in header else "jpg"
    return base64.b64decode(base64_data), file_ext


def register_student(request: HttpRequest) -> HttpResponse:
    student = None

    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        roll_no = request.POST.get("roll_no", "").strip()
        email = request.POST.get("email", "").strip()
        department = request.POST.get("department", "").strip()

        if name and roll_no:
            student, _created = Student.objects.get_or_create(
                roll_no=roll_no,
                defaults={
                    "name": name,
                    "email": email or None,
                    "department": department or None,
                },
            )
        else:
            return render(
                request,
                "attendance/register.html",
                {
                    "error": "Name and roll number are required.",
                },
                status=400,
            )

    return render(
        request,
        "attendance/register.html",
        {
            "student": student,
        },
    )


@csrf_exempt
@require_POST
def capture_face(request: HttpRequest, student_id: int) -> JsonResponse:
    """
    Receive a single captured frame (base64-encoded image) for a student
    and save it as a FaceImage. Frontend can call this multiple times
    to collect many samples.
    """
    student = get_object_or_404(Student, pk=student_id)

    try:
        data = json.loads(request.body.decode("utf-8"))
        image_data = data.get("image_data", "")
        image_bytes, file_ext = _decode_data_url(image_data)
    except Exception as exc:  # pragma: no cover - defensive
        return JsonResponse(
            {"success": False, "error": f"Could not decode image: {exc}"},
            status=400,
        )

    file_name = f"student_{student.id}_face"
    django_file = ContentFile(image_bytes, name=f"{file_name}.{file_ext}")
    face_image = FaceImage.objects.create(student=student, image=django_file)

    vec = extract_face_embedding(face_image.image.path)
    if vec is not None:
        face_image.embedding = vec.tolist()
        face_image.save(update_fields=["embedding"])
        return JsonResponse(
            {
                "success": True,
                "face_image_id": face_image.id,
                "has_embedding": True,
                "image_url": face_image.image.url,
            }
        )

    return JsonResponse(
        {
            "success": True,
            "face_image_id": face_image.id,
            "has_embedding": False,
            "image_url": face_image.image.url,
            "warning": "Image saved but no face was detected. Face the camera directly and try again.",
        }
    )


def train_model(request: HttpRequest) -> HttpResponse:
    """
    Iterate over all FaceImage records, compute embeddings where missing,
    and show a simple summary.
    """
    processed = 0
    skipped = 0
    cleared = 0
    total = FaceImage.objects.count()

    if request.method == "POST":
        for face_image in FaceImage.objects.all():
            image_path = face_image.image.path
            vec = extract_face_embedding(image_path)
            if vec is None:
                if face_image.embedding:
                    face_image.embedding = None
                    face_image.save(update_fields=["embedding"])
                    cleared += 1
                skipped += 1
                continue

            face_image.embedding = vec.tolist()
            face_image.save(update_fields=["embedding"])
            processed += 1

    students_with_counts = (
        Student.objects.annotate(num_images=Count("face_images"))
        .order_by("roll_no")
    )

    context: Dict[str, Any] = {
        "students_with_counts": students_with_counts,
        "total_images": total,
        "processed": processed,
        "skipped": skipped,
        "cleared": cleared,
    }
    return render(request, "attendance/train_model.html", context)


def manage_faces(request: HttpRequest) -> HttpResponse:
    """Browse, delete, or replace captured face images per student."""
    students = (
        Student.objects.annotate(num_images=Count("face_images"))
        .prefetch_related("face_images")
        .order_by("roll_no")
    )
    return render(request, "attendance/manage_faces.html", {"students": students})


@csrf_exempt
@require_POST
def delete_face_image(request: HttpRequest, face_image_id: int) -> JsonResponse:
    face_image = get_object_or_404(FaceImage, pk=face_image_id)
    if face_image.image:
        face_image.image.delete(save=False)
    face_image.delete()
    return JsonResponse({"success": True})


@csrf_exempt
@require_POST
def delete_student(request: HttpRequest, student_id: int) -> JsonResponse:
    """Delete a student and all related face images, embeddings, and attendance records."""
    student = get_object_or_404(Student, pk=student_id)
    name = student.name
    roll_no = student.roll_no
    image_count = student.face_images.count()
    attendance_count = student.attendance_records.count()

    for face_image in student.face_images.all():
        if face_image.image:
            face_image.image.delete(save=False)

    student.delete()

    return JsonResponse(
        {
            "success": True,
            "message": (
                f"Deleted {name} (roll {roll_no}), "
                f"{image_count} face image(s), and {attendance_count} attendance record(s)."
            ),
        }
    )


@csrf_exempt
@require_POST
def update_face_image(request: HttpRequest, face_image_id: int) -> JsonResponse:
    face_image = get_object_or_404(FaceImage, pk=face_image_id)

    try:
        data = json.loads(request.body.decode("utf-8"))
        image_data = data.get("image_data", "")
        image_bytes, file_ext = _decode_data_url(image_data)
    except Exception as exc:  # pragma: no cover - defensive
        return JsonResponse(
            {"success": False, "error": f"Could not decode image: {exc}"},
            status=400,
        )

    if face_image.image:
        face_image.image.delete(save=False)

    file_name = f"student_{face_image.student_id}_face_{face_image.id}.{file_ext}"
    face_image.image.save(file_name, ContentFile(image_bytes), save=False)

    vec = extract_face_embedding(face_image.image.path)
    face_image.embedding = vec.tolist() if vec is not None else None
    face_image.save()

    if vec is None:
        return JsonResponse(
            {
                "success": True,
                "has_embedding": False,
                "image_url": face_image.image.url,
                "warning": "Image saved but no face was detected. Run training after recapturing.",
            }
        )

    return JsonResponse(
        {
            "success": True,
            "has_embedding": True,
            "image_url": face_image.image.url,
        }
    )


def attendance_view(request: HttpRequest) -> HttpResponse:
    today = date.today()
    selected_date = today
    selected_date_param = request.GET.get("date", "").strip()
    if selected_date_param:
        try:
            selected_date = date.fromisoformat(selected_date_param)
        except ValueError:
            selected_date = today

    recent_records = (
        AttendanceRecord.objects.filter(date=selected_date)
        .select_related("student")
        .order_by("-timestamp")[:50]
    )
    context = {
        "today_date": today,
        "selected_date": selected_date,
        "recent_records": recent_records,
    }
    return render(request, "attendance/attendance.html", context)


def chatbot_page(request: HttpRequest) -> HttpResponse:
    return render(request, "attendance/chatbot.html")


@csrf_exempt
@require_POST
def chatbot_query(request: HttpRequest) -> JsonResponse:
    try:
        import json

        payload = json.loads(request.body.decode("utf-8"))
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"success": False, "error": f"Invalid payload: {exc}"}, status=400)

    message = str(payload.get("message", "")).strip()
    use_openai = bool(payload.get("use_openai", False))

    if not message:
        return JsonResponse({"success": False, "error": "Message is required."}, status=400)

    if use_openai:
        openai_answer = _try_openai_response(message)
        if openai_answer:
            return JsonResponse({"success": True, "reply": openai_answer, "engine": "openai"})

    local_answer = _build_local_chatbot_response(message)
    return JsonResponse({"success": True, "reply": local_answer, "engine": "local"})


@csrf_exempt
@require_POST
def mark_attendance(request: HttpRequest) -> JsonResponse:
    """
    Receive a short webcam frame sequence, run anti-spoof (liveness) checks,
    detect and embed the face, compare with stored embeddings, and mark
    today's attendance for the best-matching student.
    """
    try:
        import json

        data = json.loads(request.body.decode("utf-8"))
    except Exception as exc:  # pragma: no cover
        return JsonResponse(
            {"success": False, "error": f"Could not decode image: {exc}"},
            status=400,
        )

    frames_data = data.get("frames")
    image_data = data.get("image_data", "")

    import numpy as np
    import cv2  # local import: only used during attendance marking

    def _decode_data_url_to_bgr(data_url: str):
        if not isinstance(data_url, str) or not data_url.startswith("data:image"):
            return None
        _header, base64_data = data_url.split(",", 1)
        image_bytes = base64.b64decode(base64_data)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    frames_bgr: List[Any] = []
    if isinstance(frames_data, list) and frames_data:
        for item in frames_data[:12]:  # hard limit to protect request size
            img = _decode_data_url_to_bgr(str(item))
            if img is not None:
                frames_bgr.append(img)
    elif image_data:
        img = _decode_data_url_to_bgr(image_data)
        if img is not None:
            frames_bgr.append(img)

    if not frames_bgr:
        return JsonResponse(
            {"success": False, "error": "Invalid image data. Please allow camera access and try again."},
            status=400,
        )

    query_vectors: List[Any] = []
    for img in frames_bgr:
        vec = extract_face_embedding_from_bgr(img)
        if vec is not None:
            query_vectors.append(vec)

    if not query_vectors:
        return JsonResponse(
            {"success": False, "error": "No clear face detected. Please try again."},
            status=400,
        )

    labeled_vectors: List[LabeledEmbedding] = []
    for face in FaceImage.objects.exclude(embedding__isnull=True).exclude(embedding__exact=[]):
        vec = json_to_vector(face.embedding)
        labeled_vectors.append(LabeledEmbedding(student_id=face.student_id, vector=vec))

    if not labeled_vectors:
        return JsonResponse(
            {"success": False, "error": "No trained face data available. Train the model first."},
            status=400,
        )

    match = recognize_face(query_vectors, labeled_vectors)
    student_id = match.student_id
    if student_id is None:
        return JsonResponse(
            {"success": False, "error": "Face not recognized. Please make sure the student is registered and trained."},
            status=404,
        )

    student = Student.objects.get(pk=student_id)
    today = date.today()

    already_marked = AttendanceRecord.objects.filter(
        student=student,
        date=today,
        session=None,
        subject=None,
    ).exists()

    if already_marked:
        return JsonResponse(
            {
                "success": True,
                "student": {
                    "id": student.id,
                    "name": student.name,
                    "roll_no": student.roll_no,
                },
                "already_marked": True,
            }
        )

    liveness = check_liveness(frames_bgr)
    if not liveness.get("passed"):
        return JsonResponse(
            {
                "success": False,
                "error": "Liveness check failed. Please blink your eyes or move your head slightly.",
                "liveness_failed": True,
                "liveness": liveness,
            },
            status=403,
        )

    record, created = AttendanceRecord.objects.get_or_create(
        student=student,
        date=today,
        session=None,
        subject=None,
        defaults={"status": AttendanceRecord.STATUS_PRESENT},
    )

    return JsonResponse(
        {
            "success": True,
            "student": {
                "id": student.id,
                "name": student.name,
                "roll_no": student.roll_no,
            },
            "already_marked": not created,
        }
    )
    
   
