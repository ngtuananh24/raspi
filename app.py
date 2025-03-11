import cv2
import tkinter as tk
from PIL import Image, ImageTk

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
    root.after(10, update_frame)  # Cập nhật frame mỗi 10ms

cap = cv2.VideoCapture(0)  # Mở camera
root = tk.Tk()
root.title("Nhận diện biển báo")
lbl = tk.Label(root)
lbl.pack()
update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
