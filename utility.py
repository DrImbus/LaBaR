import numpy as np


def sin(x):
    return np.sin(x *np.pi/180)


    
def cos(x):
    return np.cos(x *np.pi/180)

def cosRad(x):
    return np.cos(x)
def sinRad(x):
    return np.sin(x)

def arctan(x):
    return np.arctan(x)*180/np.pi

def XRotationMatrix(angle):
    c,s = cos(angle), sin(angle)
    return np.array((
        (1,0,0),
        (0,c,-s),
        (0,s,c)
    ))


def YRotationMatrix(angle):
    c,s = cos(angle), sin(angle)
    return np.array((
        (c,0,s),
        (0,1,0),
        (-s,0,c)
    ))

def ZRotationMatrix(angle):
    c,s = cos(angle), sin(angle)
    return np.array((
        (c,-s,0),
        (s,c,0),
        (0,0,1)
    ))

def readCIF(filename):
    with open(filename) as f:
        lines = f.readlines()



        result = {}
        for i in range(len(lines)):
            lineParts = lines[i].split(" ")
            lineParts = [x.replace("\n","") for x in lineParts if x != ""]
            if lineParts[0] == '_cell_length_a':
                result['a'] = float(lineParts[1])
            if lineParts[0] == '_cell_length_b':
                result['b'] = float(lineParts[1])
            if lineParts[0] == '_cell_length_c':
                result['c'] = float(lineParts[1])
            if lineParts[0] == '_cell_angle_alpha':
                result['alpha'] = float(lineParts[1])
            if lineParts[0] == '_cell_angle_beta':
                result['beta'] = float(lineParts[1])
            if lineParts[0] == '_cell_angle_gamma':
                result['gamma'] = float(lineParts[1])
            if lineParts[0] == '_atom_site_occupancy':
                basis = []
                i+=1
                while i < len(lines):
                    
                    lineParts = lines[i].split(" ")
                    lineParts = [x.replace("\n","") for x in lineParts if x != ""]
                    basis.append(
                            {
                                lineParts[0] : [float(lineParts[3]),float(lineParts[4]),float(lineParts[5])]
                            }
                    )
                    i+=1
                    #print("basisPart: ", basis)
                #print("basis final: ", basis)

                result['basis'] = basis

    return result