import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
from main import main
import sys
import os

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TBRGS")
        self.root.geometry("1000x1000")

        self.frame = tk.LabelFrame(self.root, text="TBRGS", font=("Arial", 10))
        self.frame.pack(side="top", fill="x", padx=10)

        # Valid SCATS numbers (from load_and_process_data output, excluding 3682)
        self.choices = [
            970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127,
            3180, 3662, 3685, 3804, 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057,
            4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821
        ]

        self.originlbl = tk.Label(self.frame, text='Select Origin:')
        self.originlbl.grid(row=0, column=0, padx=3)
        self.origin_dd = ttk.Combobox(self.frame, values=self.choices)
        self.origin_dd.grid(row=1, column=0, padx=3)

        self.destlbl = tk.Label(self.frame, text='Select Destination:')
        self.destlbl.grid(row=2, column=0, padx=3)
        self.dest_dd = ttk.Combobox(self.frame, values=self.choices)
        self.dest_dd.grid(row=3, column=0, padx=3)

        self.MLlbl = tk.Label(self.frame, text='Select ML Model:')
        self.MLlbl.grid(row=0, column=1, padx=3)
        self.ML_dd = ttk.Combobox(self.frame, values=["LSTM", "GRU", "FNN"])
        self.ML_dd.grid(row=1, column=1, padx=3)

        self.Timelbl = tk.Label(self.frame, text='Select Time:')
        self.Timelbl.grid(row=2, column=1, padx=3)
        self.Time_dd = ttk.Combobox(self.frame, values=[
            "00:00", "00:15", "00:30", "00:45", "01:00", "01:15", "01:30", "01:45",
            "02:00", "02:15", "02:30", "02:45", "03:00", "03:15", "03:30", "03:45",
            "04:00", "04:15", "04:30", "04:45", "05:00", "05:15", "05:30", "05:45",
            "06:00", "06:15", "06:30", "06:45", "07:00", "07:15", "07:30", "07:45",
            "08:00", "08:15", "08:30", "08:45", "09:00", "09:15", "09:30", "09:45",
            "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45",
            "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
            "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45",
            "16:00", "16:15", "16:30", "16:45", "17:00", "17:15", "17:30", "17:45",
            "18:00", "18:15", "18:30", "18:45", "19:00", "19:15", "19:30", "19:45",
            "20:00", "20:15", "20:30", "20:45", "21:00", "21:15", "21:30", "21:45",
            "22:00", "22:15", "22:30", "22:45", "23:00", "23:15", "23:30", "23:45"
        ])
        self.Time_dd.grid(row=3, column=1, padx=3)

        self.ConfirmBtn = tk.Button(self.frame, text="Run", font=("Arial", 10), command=self.RunMain)
        self.ConfirmBtn.grid(row=4, column=1, padx=3)

        # Frame for image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill="both", expand=True)

        # Initialize with default image
        self.default_img_path = "default_map.png"
        self.original_img = Image.open(self.default_img_path) if os.path.exists(self.default_img_path) else None
        self.tk_img = None

        self.resultImglbl = tk.Label(self.image_frame)
        self.resultImglbl.pack(fill="both", expand=True)

        # Bind resize event
        self.image_frame.bind("<Configure>", self.resize_image)

        self.result_text = tk.Text(self.frame, height=9, width=50)
        self.result_text.grid(row=0, column=2, rowspan=5, padx=3)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def RunMain(self):
        origin = self.origin_dd.get()
        dest = self.dest_dd.get()
        file_path = "Scats Data October 2006.xls"
        ml = self.ML_dd.get()
        time = self.Time_dd.get()

        # Input validation
        if not origin or not dest:
            messagebox.showwarning("Warning", "Please select origin and destination")
            return
        if not ml:
            messagebox.showwarning("Warning", "Please select a model")
            return
        if origin == dest:
            messagebox.showwarning("Warning", "Origin and destination cannot be the same")
            return
        if not time:
            messagebox.showwarning("Warning", "Please select a time")
            return

        try:
            # Convert origin and destination to integers
            origin = int(origin)
            dest = int(dest)

            # Run main function
            paths = main(file_path, origin, dest, ml, time)
            
            # Display paths in GUI
            self.result_text.delete(1.0, tk.END)
            if not paths:
                self.result_text.insert(tk.END, "No valid paths found.")
            else:
                for i, path_info in enumerate(paths, 1):
                    path = path_info['path']
                    travel_time = path_info['travel_time']
                    self.result_text.insert(tk.END, f"Route {i}: {' -> '.join(map(str, path))}\n")
                    self.result_text.insert(tk.END, f"Travel time: {travel_time:.2f} minutes\n\n")
            
            # Load traffic network image
            img_path = "traffic_network.png"
            if os.path.exists(img_path):
                self.original_img = Image.open(img_path)
                self.resize_image()
            else:
                messagebox.showwarning("Warning", "Traffic network image not found")
                self.original_img = Image.open(self.default_img_path) if os.path.exists(self.default_img_path) else None
                self.resize_image()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def resize_image(self, event=None):
        # Get frame's dimensions
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()

        # Prevent errors if image is not found or frame is too small
        if not self.original_img or frame_width <= 0 or frame_height <= 0:
            return
        
        # Scale image while preserving aspect ratio
        img_width, img_height = self.original_img.size
        scale = min(frame_width / img_width, frame_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_img = self.original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_img)

        # Update image label
        self.resultImglbl.config(image=self.tk_img)
        self.resultImglbl.image = self.tk_img
    
    def close(self):
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    GUI()