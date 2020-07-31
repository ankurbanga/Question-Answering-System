import subprocess
from tkinter import *


tk=Tk()
tk.title("QA System")
ans="Ask a Question"

w = Label(tk,
    text=ans,
    foreground="black", 
    background="teal",  
    font=("Helvectica",42),
    width=50,
    height=10,
    wraplength=1000,
    justify=LEFT
)
w.pack()

ent=Entry(tk,
         width=50,
         font=("Arial",30))
ent.pack()
def show_data():
    ans=ent.get()
    p = subprocess.Popen(["python3","Test.py", ans], stdout=subprocess.PIPE)
    w["text"] =p.communicate()[0]


Button(tk,
      text="Find Answer",
      command=show_data,
      width=30,
      height=4).pack()

tk.mainloop()
