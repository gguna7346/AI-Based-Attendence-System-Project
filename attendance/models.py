from django.db import models


class Student(models.Model):
    roll_no = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    department = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["roll_no"]

    def __str__(self) -> str:
        return f"{self.roll_no} - {self.name}"


class FaceImage(models.Model):
    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name="face_images"
    )
    image = models.ImageField(upload_to="students/%Y/%m/%d/")
    # Optional precomputed embedding for faster matching; can be filled during training
    embedding = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"FaceImage for {self.student} at {self.created_at}"


class AttendanceRecord(models.Model):
    STATUS_PRESENT = "present"
    STATUS_ABSENT = "absent"
    STATUS_LATE = "late"

    STATUS_CHOICES = [
        (STATUS_PRESENT, "Present"),
        (STATUS_ABSENT, "Absent"),
        (STATUS_LATE, "Late"),
    ]

    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name="attendance_records"
    )
    date = models.DateField()
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_PRESENT
    )
    session = models.CharField(max_length=100, blank=True, null=True)
    subject = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        ordering = ["-date", "-timestamp"]
        unique_together = ("student", "date", "session", "subject")

    def __str__(self) -> str:
        return f"{self.student} - {self.date} - {self.status}"

