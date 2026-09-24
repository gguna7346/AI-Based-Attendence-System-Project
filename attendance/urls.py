from django.urls import path

from . import views

app_name = "attendance"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("register/", views.register_student, name="register_student"),
    path("capture-face/<int:student_id>/", views.capture_face, name="capture_face"),
    path("train-model/", views.train_model, name="train_model"),
    path("manage-faces/", views.manage_faces, name="manage_faces"),
    path("face-image/<int:face_image_id>/delete/", views.delete_face_image, name="delete_face_image"),
    path("student/<int:student_id>/delete/", views.delete_student, name="delete_student"),
    path("face-image/<int:face_image_id>/update/", views.update_face_image, name="update_face_image"),
    path("attendance/", views.attendance_view, name="attendance"),
    path("chatbot/", views.chatbot_page, name="chatbot"),
    path("chatbot-query/", views.chatbot_query, name="chatbot_query"),
    path("mark-attendance/", views.mark_attendance, name="mark_attendance"),
]

