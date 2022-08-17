import numpy as np
import open3d as o3d 
import math
from matplotlib import pyplot as plt

RangeImg1 = np.zeros( (100 , 360) )
RangeImg2 = np.zeros( (100 , 360) )

Mask1 = np.zeros( (100 , 360) )
Mask2 = np.zeros( (100 , 360) )


ObjectLabels = [4,5,6,8,9,10,11,12,13,14]

c= math.cos
s = math.sin

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

def ShiftCircular( MaskedRange2, NumShift ):

	NewImg2 = np.zeros( (100 , 360) )

	for i in range( 360 ):

		newcol = (i+NumShift)%360 

		NewImg2[:, newcol ] = MaskedRange2[:, i ]


	return NewImg2


def GetCost( Img1, Img2 ):

	cost = -np.linalg.norm(Img1 -Img2 , axis=0)
	cost = np.sum(cost)

	return cost


def Optimization( MaskedRange1 , MaskedRange2 ):

	steps = np.arange(0, 359 , 1)
	AllCost = []

	for step in steps:

		NewImg2 = ShiftCircular( MaskedRange2 , step)
		cost = GetCost(MaskedRange1 , NewImg2)
		AllCost.append(cost)

	OptimalYaw = np.argmin(AllCost)

	return OptimalYaw*3.14/180


def VisualizePCD( PCD1 , PCD2  ):

	PointCloud1  =o3d.geometry.PointCloud()
	PointCloud2  =o3d.geometry.PointCloud()

	PointCloud1.points = o3d.utility.Vector3dVector(PCD1)
	PointCloud2.points = o3d.utility.Vector3dVector(PCD2)

	PointCloud1.paint_uniform_color([1, 0.706, 0])
	PointCloud2.paint_uniform_color([0.706, 0.706, 0.5])

	o3d.visualization.draw_geometries([PointCloud1, PointCloud2])



[ SegPCD1, PCD1], [SegPCD2 , PCD2] = LoadPCD()


VisualizePCD(PCD1, PCD2)
Mask1, Mask2= GetMask([ SegPCD1, PCD1], [SegPCD2 , PCD2])

MaskedRange1, MaskedRange2  = GetMaskedRangeImage(Mask1, Mask2, PCD1, PCD2)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.imshow(MaskedRange1)
ax2 = fig.add_subplot(2,1,2)
ax2.imshow(MaskedRange2)

plt.show()

OptimalYaw = Optimization(MaskedRange1, MaskedRange2)

R = np.array( [ [ c(OptimalYaw), -s(OptimalYaw) , 0] , 
				[ s(OptimalYaw) , c(OptimalYaw) ,0 ] ,
				[0 , 0 , 1] ]  )


TempPointCloud  =o3d.geometry.PointCloud()
TempPointCloud.points = o3d.utility.Vector3dVector(PCD2)

TempPointCloud = TempPointCloud.rotate(R.T , center=(0, 0, 0 ))

AugumentedPCD = np.asarray(TempPointCloud.points)

VisualizePCD(PCD1, AugumentedPCD)

