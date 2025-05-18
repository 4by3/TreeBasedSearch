
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
from main import main


#How to use Tkinter
#There is a root window defined as tk.Tk(), all widgets are placed into this, as such make sure when creating a new label, dropdown menu, etc. the first parameter should be the window it is placed in
#use [widget].pack() to place the widget onto the window, note, can modify x/y padding and also the specific side
#another way to place widgets is using .place() but place() uses exact x/y coords & height/width pixels and doesnt automatically move along with window when it is expanded.
#can also use .grid() to place stuff
#can also use .frame to make a 'window' within which can better place shit, instead of putting the root window as the first param, put this frame guy instead.

#TO DO:
#1. Add picture of the map
#2. Maybe live animation of a map to highlight currently selected nodes?
#3. Add functionality to FNN
#4. Need to add time functionality
#5. Maybe put output into GUI window ffs (Integrate pic? & also list down routes on the window)


class GUI:

    def __init__(self):
        
        #state to check if the program was run.
        self.ran = False

        #makes window, 1000x1000 pixels by default
        self.root = tk.Tk()
        self.root.title("TBRGS")
        self.root.geometry("1000x1000")

        #Title
        self.title = tk.Label(self.root, text = "TBRGS", font = ("Arial", 30))
        self.title.pack(side = "top", padx = 20, pady= 20)

        #all nodes
        self.choices = [4063, 2820, 3682, 3002, 3180, 2200, 4264, 3812, 4272, 4263, 2846, 4821, 3662, 4270, 4030, 2825, 3126, 2000, 4273, 4812, 2827, 4035, 4262, 4324, 970, 3122, 4034, 4051, 4057, 3127, 3685, 4043, 3001, 4321, 4032, 4040, 3804, 3120, 4335, 4266]

        #Dropdown menu for origin
        self.originlbl = tk.Label(self.root, text = 'Select Origin:')
        self.originlbl.pack()
        self.origin_dd = ttk.Combobox(self.root, values = self.choices)
        self.origin_dd.pack()

        #Dropdown menu for destination
        self.destlbl = tk.Label(self.root, text = 'Select Destination:')
        self.destlbl.pack()
        self.dest_dd = ttk.Combobox(self.root, values = self.choices)
        self.dest_dd.pack()

        #Dropdown menu for MLmodel selection
        self.MLlbl = tk.Label(self.root, text = 'Select ML Model:')
        self.MLlbl.pack()
        self.ML_dd = ttk.Combobox(self.root, values = ["LSTM", "GRU", "FNN"])
        self.ML_dd.pack()

        #Dropdown menu for time, not done yet
        self.Timelbl = tk.Label(self.root, text = 'Select Time:')
        self.Timelbl.pack()
        self.Time_dd = ttk.Combobox(self.root, values = ["Yeah", "Nah", "Pls", "Add", "Functionality"])
        self.Time_dd.pack()


        #Confirmation button to submit form.
        self.ConfirmBtn = tk.Button(self.root, text = "Run", font = ("Arial", 15), command = self.RunMain)
        self.ConfirmBtn.pack( side = "bottom", padx = 20, pady = 100)

    

        self.resultImglbl = tk.Label(self.root)
        self.resultImglbl.pack(side= "bottom", pady= 20 )



        self.root.mainloop()

    
    def RunMain(self):
    #Runs main(file_path, origin, dest, ml model)        
        origin = self.origin_dd.get()
        dest = self.dest_dd.get()
        file_path = "Scats Data October 2006.xls"
        ml = self.ML_dd.get()

    #Error checking for values in dropdown menu.
        if not origin or not dest:
            messagebox.showwarning("Warning", "Please select origin/destination")
            return
        
        if not ml:
            messagebox.showwarning("Warning", "Please select a model:")
            return

        if origin == dest:
            messagebox.showwarning("Warning", "Origin & Destination are the same")
            return
        

        main(file_path, origin, dest, ml)

        #updates image
        self.showImage()

        return

    def showImage(self):
        #updating the image label in the mainloop to show new result image
        resultImg = Image.open("traffic_network.png").resize((600,600))
        resultImg = ImageTk.PhotoImage(resultImg)

        self.resultImglbl.config(image=resultImg)
        self.resultImglbl.image = resultImg

        return

            
GUI()