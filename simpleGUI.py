from tkinter import *

root = Tk()

e = Entry(root, width=80)
e.grid(row=0, column=0, columnspan=2)  # Adjust grid placement
e.insert(0,"Enter here : ")
#e.pack()

def myclick():
    mylabel = Label(root, text="Hello " + e.get())
    mylabel.grid(row=2, column=0, columnspan=2)  # Use grid here

mybutton = Button(root, text="Show", command=myclick)
mybutton.grid(row=1, column=1, columnspan=2)  # Use grid here

root.mainloop()

