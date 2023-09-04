#%%
import numpy as np
import pandas as pd
from math import sqrt, exp

import matplotlib.pyplot as plt

from utility import (
    sin,
    cos,
    cosRad,
    sinRad,
    arctan,
    XRotationMatrix,
    YRotationMatrix,
    ZRotationMatrix,
    readCIF
)

PLANCK = 6.62607*10**(-34)
C = 299792458
ELEMENTARY_CHARGE = 1.60217663*10**(-19)


########################################################################

#                               USER INPUT

########################################################################
        
importData = True
if importData:
    crystalData = readCIF("Sr2RuO4.cif")
    a = crystalData['a']
    b = crystalData['b']
    c = crystalData['c']
    alpha = crystalData['alpha']
    beta = crystalData['beta']
    gamma = crystalData['gamma']
    basis_atoms = crystalData['basis']
else:
    a = 5
    b = 6
    c = 7
    alpha = 90
    beta = 90
    gamma = 90

    #basis_atoms
    basis_atoms = [
        {'C':[0,0,0]},
        {'O2-':[0.5,0.5,0.5]},
        {'Sr2+':[0.8,0.3,0.1]}
    ]
print(a)
print(b)
print(c)
print(alpha)
print(beta)
print(gamma)
for eleme in basis_atoms:
    print(eleme)
        

#laboratory parameters
d_screen = 30 #distance between screen and cystal in mm
w_screen = 113 #screen width in mm
h_screen = 143 #screen height in mm
aparture_radius = 5 #radius of the aparture (a cutout) in the middel that doesnt receive spots

#euler-angles of the crystal in degrees
x_rotation = 0
y_rotation = -10
z_rotation = -10

#array of rotation matrices that will be applied to the crystal.
#The order of rotations Z->X->Y is equivilant to CLIP so it is not recommended to change the order.
#You can add additional rotations after the YRotationMatrix to adjust the image:
#ZRotationMatrix => The Laue Image gets rotated in the negative x-direction
#YRotationMatrix => The Laue Image gets rotated in the negative y-direction
#XRotationMatrix => The Laue Image gets rotated counter-clockwise around the center
rotations = [
    ZRotationMatrix(z_rotation),
    XRotationMatrix(x_rotation),
    YRotationMatrix(y_rotation)
]


#this contains the parameters used to calculate the Gauss-Aproximation
#of the atomic form factor.
#The parameters where taken from the international tables of crystallography vol. C
form_factor_params = pd.read_csv("form_factor_parameters.csv", sep=";")

#defines the maximum sum of laue-indices
#a higher maxHKL produces a more accurate image but also slows down the programm
maxHKL = 20

U_a = 20  #acceleration Voltage of the X-Ray tube in kV
lambda_min = PLANCK * C / ( ELEMENTARY_CHARGE * U_a * 1000)*10**10


#contrast function contrast: [0,1] -> [0,1]
#it is applied only to the plot 
#not to the printed out csv-file
def contrast(x):
    #return x**(1/4)
    handles = [
        (0,0),
        (0.1,0.9),
        (1,1)
        ]
    x = x**(1/2)

    for i in range(len(handles)-1):
        a = handles[i]
        b = handles[i+1]
        if x >= a[0] and x <= b[0]:
            distanceBetweenAB = (x-a[0])/(b[0]-a[0])
            return (b[1]-a[1])*distanceBetweenAB + a[1]
    return x



#relationship between the radius of the spots and their intensity
def sizeFunction(x):
    minV = 0.1
    steepness = 3
    maxIncrease = 2*minV
    maxV = 10*minV
    result = maxIncrease * exp(x*steepness)/exp(steepness) + minV
    result = min(result, maxV)
    return result*4.5

########################################################################

#                               FUNCTIONS

########################################################################


def xray_spectrum(wavelength):
    kramer = lambda x : (x/lambda_min - 1) * 1/(x**3)
    result = kramer(wavelength)/kramer(1.5*lambda_min)
    if result < 0:
        return 0
    return result
    

def angle_between_vectors(v1,v2):
    v1 = v1/sqrt(v1.dot(v1)) #normalizing
    v2 = v2/sqrt(v2.dot(v2)) #normalizing
    return np.arccos( v1.dot(v2) ) * 180/np.pi

def interception_screen(vector):

    # normal = np.array((1,0,0))
    # point = np.array((-d_screen,0,0))
    # t = point.dot(normal)/vector.dot(normal)

    # result = vector*t
    # result[1] *= -1

    if vector[0] == 0:
        return -1
    
    t = d_screen/vector[0]

    return np.array((-t*vector[1],t*vector[2]))




########################################################################

#           CALCULATING RECIPROCAL BASIS VECTORS IN LAB SYSTEM

########################################################################


real_basis_a = np.array((a,0,0))

real_basis_b = np.array((cos(gamma)*b,sin(gamma)*b,0))
real_basis_c = np.array((cos(beta)*c, 
                        cos(alpha)*sin(gamma)*c,
                        sqrt(1-cos(beta)**2-(cos(alpha)*sin(gamma))**2 )*c))




for r in rotations:
    real_basis_a = r.dot(real_basis_a)
    real_basis_b = r.dot(real_basis_b)
    real_basis_c = r.dot(real_basis_c)



volume = np.cross(real_basis_a,real_basis_b).dot(real_basis_c)

rec_basis_a = np.cross(real_basis_b, real_basis_c)*1/volume*2*np.pi
rec_basis_b = np.cross(real_basis_c, real_basis_a)*1/volume*2*np.pi
rec_basis_c = np.cross(real_basis_a, real_basis_b)*1/volume*2*np.pi

#return the reciprocal lattice vector with components hkl
def getG(h,k,l):
    return h*rec_basis_a + k * rec_basis_b + l * rec_basis_c

#return the distance between the miller-planes that G_hkl is normal to
def getD(h,k,l):
    G = getG(h,k,l)
    G_magnitude = sqrt(G.dot(G))
    return np.pi*2/G_magnitude




########################################################################

#                   CALCULATE MILLER INDICES

########################################################################

def getHKLOfSum(sum = 1):
    if sum < 1:
        return[]
    results = []
    for a in range(sum,-1,-1):
        for b in range(sum-a,-1,-1):
            c = sum-a-b
            results.append((a,b,c))

            if(a != 0):
                results.append((-a,b,c))
            
            if(b != 0):
                results.append((a,-b,c))
            
            if(c != 0):
                results.append((a,b,-c))
            
            if(a != 0 and b!=0):
                results.append((-a,-b,c))
            
            if(a != 0 and c!= 0):
                results.append((-a,b,-c))
            
            if(b != 0 and c!=0):
                results.append((a,-b,-c))
            
            if(a != 0 and b != 0 and c!= 0):
                results.append((-a,-b,-c))
    return results

def getHKL(until_sum = 1):
    results = []
    for i in range(until_sum+1):
        results= results + getHKLOfSum(i)
    return results


########################################################################

#                   CALCULATE STRUCTURE FACTOR

########################################################################



def getAtomicFormFactor(G, atomType):

    a1 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'a1'].iloc[0]
    a2 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'a2'].iloc[0]
    a3 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'a3'].iloc[0]
    a4 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'a4'].iloc[0]
    b1 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'b1'].iloc[0]
    b2 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'b2'].iloc[0]
    b3 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'b3'].iloc[0]
    b4 = form_factor_params.loc[form_factor_params['Element'] == atomType, 'b4'].iloc[0]
    c = form_factor_params.loc[form_factor_params['Element'] == atomType, 'c'].iloc[0]

    result = c
    result += a1*exp(-b1 * ( G/ (4*np.pi) )**2)
    result += a2*exp(-b2 * (G/ (4*np.pi) )**2)
    result += a3*exp(-b3 * (G/ (4*np.pi) )**2)
    result += a4*exp(-b4 * (G/ (4*np.pi) )**2)

    return result

def getAbsoluteStructureFactor(h,k,l):
    result = 0
    realPart = 0
    imPart = 0

    rlv = getG(h,k,l)
    G = sqrt(rlv.dot(rlv)) #G in 1/Angstrom
    for atom in basis_atoms:
        for type, position in atom.items():

            realPart += getAtomicFormFactor(G,type) * cosRad(
                2*np.pi*(
                h*position[0]+k*position[1]+l*position[2]
                )
            )

            imPart -= getAtomicFormFactor(G,type) * sinRad(
                2*np.pi*(
                h*position[0]+k*position[1]+l*position[2]
                )
            )


    result = sqrt(realPart**2+imPart**2)
    return result

########################################################################

#                   CALCULATE LAUE BACK REFLECTION PATTERN

########################################################################


#Step 1: calculte all reciprocal lattice vectors that 
#        can contribute to reflections of the screen
#        => filter rlv's by angles

#all miller indices
millerIndices = getHKL(maxHKL)

#all reciprocal lattice vectors
reciprocalLatticeVectors = [getG(hkl[0],hkl[1],hkl[2]) for hkl in millerIndices]

anglesXAxis = [angle_between_vectors(np.array((1,0,0)), rec) for rec in reciprocalLatticeVectors]

filteredReciprocalLatticeVectors = []
filteredMillerIndices = []
minAngle = 1/2 * arctan(aparture_radius/d_screen)
maxAngle = 1/2*arctan(sqrt(w_screen**2+h_screen**2)/d_screen)

#filter the reciprocal lattice vectors by the angle 
#they enclose with the x-axis
for i in range(len(reciprocalLatticeVectors)):
    #rlv's with smaller angles will reflect into the aperture
    minAngle = 1/2 * arctan(aparture_radius/d_screen)
    #rlv's with greater angles will reflect outside of the screen
    maxAngle = 1/2*arctan(sqrt(w_screen**2+h_screen**2)/d_screen)
    if minAngle < anglesXAxis[i] and anglesXAxis[i] < maxAngle:
        filteredReciprocalLatticeVectors.append(reciprocalLatticeVectors[i])
        filteredMillerIndices.append(millerIndices[i])

reciprocalLatticeVectors = filteredReciprocalLatticeVectors
millerIndices = filteredMillerIndices




#calculate the position of the spots on the screen 
reflectedVectors = []
angles = []
spots = []

for rlv in reciprocalLatticeVectors:
    incomingBeam = np.array((-1,0,0))
    reflected = incomingBeam - 2*(rlv.dot(incomingBeam))*rlv/(rlv.dot(rlv))
    reflectedVectors.append(reflected)
    spots.append(interception_screen(reflected))

    angle = 90 - angle_between_vectors(reflected,rlv)
    angles.append(angle)


wavelengths = []
planeDistances = []
distanceTravelled = []

#calculate the wavelength of the reflected beams

for i in range(len(millerIndices)):
    
    planeDistance = getD(millerIndices[i][0],millerIndices[i][1],millerIndices[i][2])
    wavelength = 2*planeDistance*sin(angles[i])
    d = sqrt(spots[i].dot(spots[i]) + d_screen**2)

    wavelengths.append(wavelength)
    planeDistances.append(planeDistance)
    distanceTravelled.append(d)


#calculate the intensity
intensities = []
structureFactors = []
max_intensity = 0


for i in range(len(spots)):
    temp_intensity = 1




    #apply spectrum
    temp_intensity *= xray_spectrum(wavelengths[i])     

    

    #lorentz factor
    temp_intensity*= wavelengths[i]**4/sin(angles[i])**2

    #polarization factor
    temp_intensity*= (1+cos(angles[i]*2)**2)/2



    h,k,l = millerIndices[i][0],millerIndices[i][1],millerIndices[i][2]

    #structure factor
    F = getAbsoluteStructureFactor(h,k,l)
    temp_intensity*=F**2


    if temp_intensity > max_intensity:
        max_intensity = temp_intensity

    structureFactors.append(F)
    intensities.append(temp_intensity)






reflections = pd.DataFrame({
    'hkl':millerIndices,
    'x':[s[0] for s in spots],
    'y':[s[1] for s in spots],
    '2 Theta': [2*angle for angle in angles],
    'wavelength':wavelengths,
    'd_hkl':planeDistances,
    'G': [sqrt(rlv.dot(rlv)) for rlv in reciprocalLatticeVectors],
    'F': structureFactors,
    'intensities':intensities
})


reflections = reflections.loc[reflections['intensities'] != 0]
reflections = reflections[abs(reflections['x'])<w_screen/2]
reflections = reflections[abs(reflections['y'])<h_screen/2]



maxIntensity = reflections['intensities'].max()
reflections['intensities'] = reflections['intensities']/maxIntensity




reflections.to_csv("reflectionInfo.csv", index=False, sep=";" )

#reflections.head()


#plt.figure(figsize=(w_screen,h_screen))

axes = plt.axes()
axes.set_facecolor("black")

#plt.axis("equal")

plt.scatter(reflections['x'], 
            reflections['y'],
            s = [sizeFunction(contrast(i)) for i in reflections['intensities']],
            c=(1,1,1),
            alpha = [contrast(i) for i in reflections['intensities']],
            linewidths=0
            )
plt.plot([-w_screen/2,-w_screen/2,w_screen/2,w_screen/2,-w_screen/2],
         [-h_screen/2,h_screen/2,h_screen/2,-h_screen/2,-h_screen/2]
         ,c=(1,0,0)
         ,linewidth = 0.1
         )

plt.plot(
    [aparture_radius * np.cos(t) for t in np.linspace(0,np.pi*2,50)],
    [aparture_radius * np.sin(t) for t in np.linspace(0,np.pi*2,50)],
    c=(1,0,0)
    ,linewidth = 0.1
)


# plt.axis([-w_screen/2, w_screen/2, -h_screen/2, h_screen/2], aspect = 'equal')
# plt.gca().set_aspect('equal')
#plt.xlim((-w_screen/2* 1.1, w_screen/2 * 1.1))
plt.ylim((-h_screen/2* 1.1, h_screen/2 * 1.1))
plt.gca().set_aspect('equal')
#plt.figure(figsize=(5,8))
plt.savefig("figure1.png",dpi=600)
plt.show()








# %%
