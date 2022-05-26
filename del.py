import os 

def main():
    path = "C:/Users/JSO21/Downloads/sharp"
    a = os.listdir(path)
    ier = 0
    for i in a:
        print(i)
        if not i.find("img"):
            continue
        os.rename(f"{path}/{i}", f"{path}/img_{ier}.jpg")
        ier+=1

main()