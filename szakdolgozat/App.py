from UserInterface.UI import UI
from BusinessLogic.BL import BL
from DataBase.DB import DB

if __name__ == "__main__":
    database = DB()
    business_logic = BL(database)
    view = UI(business_logic)
    view.window.mainloop()
    
# 