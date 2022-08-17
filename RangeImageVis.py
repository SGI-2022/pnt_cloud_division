import numpy as np
import open3d as o3d 
import math
from matplotlib import pyplot as plt

RangeImg1 = np.zeros( (100 , 360) )
RangeImg2 = np.zeros( (100 , 360) )

Mask1 = np.zeros( (100 , 360) )
Mask2 = np.zeros( (100 , 360) )


ObjectLabels = [4,5,6,8,9,10,11,12,13,14]


def LoadPCD():

	SegPCD1 = np.load("WaymoExamples/0084_seg.npy")[:, 1]
	PCD1 = np.load( "WaymoExamples/0084.npy")[:len(SegPCD1), :3]

	SegPCD2 = np.load("WaymoExamples/0025_seg.npy")[:, 1]
	PCD2 = np.load( "WaymoExamples/0025.npy")[:len(SegPCD2), :3]


	return [ SegPCD1, PCD1], [SegPCD2 , PCD2]

def ConvertToSpherical( Cart ):

	# print(Cart)

	r = np.linalg.norm(Cart)

	phi = math.atan2( Cart[1] ,Cart[0])*180/3.14

	theta = math.acos( Cart[2]/ r )*180/3.14

	Sph = np.asarray([ r,theta, phi])

	return Sph

def GetMask( Data1, Data2 ):

	[ SegPCD1, PCD1], [SegPCD2 , PCD2] = Data1, Data2

	for i in range(len(PCD1)):
		Sph =  ConvertToSpherical( PCD1[i] )

		if( SegPCD1[i] in ObjectLabels ):
			Mask1[ int(Sph[1])][int(Sph[2])] = 1 

	for i in range(len(PCD2)):
		Sph =  ConvertToSpherical( PCD2[i] )

		if( SegPCD2[i] in ObjectLabels ):
			Mask2[ int(Sph[1])][int(Sph[2])] = 1 

	return Mask1, Mask2

def GetMaskedRangeImage( Mask1, Mask2, PCD1 , PCD2 ):

	for i in range(len(PCD1)):
		Sph =  ConvertToSpherical( PCD1[i] )

		RangeImg1[ int(Sph[1])][int(Sph[2])] = Sph[0] #Sph[0]

	for i in range(len(PCD2)):
		Sph =  ConvertToSpherical( PCD2[i] )
		RangeImg2[ int(Sph[1])][int(Sph[2])] = Sph[0] #Sph[0]	


	MaskedRange1 = np.multiply(Mask1 ,  RangeImg1)
	MaskedRange2 = np.multiply(Mask2 ,  RangeImg2)

	return MaskedRange1, MaskedRange2 



[ SegPCD1, PCD1], [SegPCD2 , PCD2] = LoadPCD()

Mask1, Mask2= GetMask([ SegPCD1, PCD1], [SegPCD2 , PCD2])

MaskedRange1, MaskedRange2  = GetMaskedRangeImage(Mask1, Mask2, PCD1, PCD2)

