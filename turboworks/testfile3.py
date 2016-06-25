from testfile import someClass
#testing concepts to see why optimize_task is having importing problems
#this is the main file
x=someClass('testfile2')

class Base(object):
    def __init__(self):
        print "Base created"

class ChildA(Base):
    def __init__(self):
        print "no inheritance"

class ChildB(Base):
    def __init__(self):
        super(ChildB, self).__init__()

ChildA()