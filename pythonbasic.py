class foo:
    def __init__(self, name):
        self.name = name
        print("HELLO ,[%s]" %(self.name))

    def boo(self, letter = False):
        if letter :
            print("BOO [%s]" %(self.name.upper()))

        else:
            print('boo [%s]' %(self.name))

print("Class defined")


f = foo('Seungjae')


f.boo()
f.boo(letter = True)
