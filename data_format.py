
# coding: utf-8

# In[1]:

import struct
from PIL import Image



# In[2]:

size_of_data = 8199
size_of_batch = 4780


# In[3]:

def read_record_ETL8G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)


# In[4]:


FILE_HEAD = "ETL8G/ETL8G_"
for n in range(32):
    digits = "{0:02d}".format(n + 1)
    file_name = FILE_HEAD + digits

    with open(file_name, "rb") as f:
        for i in range(size_of_batch):
            f.seek(i * size_of_data)
            data = read_record_ETL8G(f)
            # print( data[0:-2], hex(data[1]), hex(data[1])[-4:])
            iE = Image.eval(data[-1], lambda x: 255 - x * 16)
            fn = 'ETL8G_{:d}_{:s}.png'.format(data[0], hex(data[1])[-4:])
            iE.save("data_all/" + fn, 'PNG')
            # print("data No %3d colleted" % i)
    print("FILE NO:%2d finished" % n)






