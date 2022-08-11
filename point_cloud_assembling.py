import numpy as np 
from visualize import visualize_point_cloud_colored
from waymo_data import is_foreground
from random import shuffle


def assembling_helper(id1, id2):
    labels1 = np.load("WaymoExamples/"+id1+"_seg.npy")
    seg_labels1 = labels1[:, 1]
    # group_id1 = labels1[:, 2:]

    labels2 = np.load("WaymoExamples/"+id2+"_seg.npy")
    seg_labels2 = labels2[:, 1]

    pc1 = np.load("WaymoExamples/"+id1+".npy")[:len(labels1), :3]
    pc2 = np.load("WaymoExamples/"+id2+".npy")[:len(labels2), :3]

    store1 = {}
    for i in range(len(seg_labels1)):
        label = seg_labels1[i]
        if(label not in store1):
            store1[label] = []
        store1[label].append(pc1[i])

    mean1 = {}
    for instance in store1:
        mean1[instance] = np.mean(store1[instance], axis=0)

    store2 = {}
    for i in range(len(seg_labels2)):
        label = seg_labels2[i]
        if(label not in store2):
            store2[label] = []
        store2[label].append(pc2[i])

    mean2 = {}
    for instance in store2:
        mean2[instance] = np.mean(store2[instance], axis=0)
    
    ### distance 
    radius1 = {}
    for instance in store1:
        rad = 0
        center  = mean1[instance]
        for i in range(len(store1[instance])):
            pt = store1[instance][i]
            diff = center - pt
            dist = np.sum(diff**2)**(0.5)
            rad = max(rad, dist)

        radius1[instance] = rad

    radius2 = {}
    for instance in store2:
        rad = 0
        center  = mean2[instance]
        for i in range(len(store2[instance])):
            pt = store2[instance][i]
            diff = center - pt
            dist = np.sum(diff**2)**(0.5)
            rad = max(rad, dist)

        radius2[instance] = rad

    mixed_scene = set()
    
    for instance1 in store1:
        dummy_var = 1
        for instance2 in store2:
            dist_rad = radius1[instance1] + radius2[instance2]
            diff = mean1[instance1] - mean2[instance2]
            dist_centers = np.sum(diff**2)**(0.5)
            if(dist_centers < dist_rad): # they overlap
                dummy_var = 0
                random = np.random.randint(0,2)
                chosen_instance = instance1 if random == 0 else instance2
                print(random, chosen_instance)
                mixed_scene.add((random, chosen_instance))
                break
        if(dummy_var == 1):
            mixed_scene.add((2, instance1))
    
    # for instance2 in store2:
    #     dummy_var = 1
    #     for instance1 in store1:
    #         dist_rad = radius1[instance1] + radius2[instance2]
    #         diff = mean1[instance1] - mean2[instance2]
    #         dist_centers = np.sum(diff**2)**(0.5)
    #         if(dist_centers < dist_rad): # they overlap
    #             dummy_var = 0
    #             random = np.random.randint(0,2)
    #             chosen_instance = instance1 if random == 1 else instance2
    #             mixed_scene.add((random, chosen_instance))
    #             break
    #     if(dummy_var == 1):
    #         mixed_scene.add((2, instance1))
    results = np.array([]).astype(int)
    color_labels = []
    for scene, instance in mixed_scene:
        points = store1[instance] if scene == 0 else store2[instance]
        color_labels += [2 for i in range(len(points))]
        results = np.concatenate(points, results).astype(int)
    colors = np.array([
		[0.7, 0.5, 0.1],
		[0.1, 0.3, 0.6],
		[0.9, 0.7, 0.7]
	])

    visualize_point_cloud_colored(results, color_labels, colors)
    # print(color_labels)




    # print(radius2, '\n', radius1, '\n', mixed_scene)
    
    return None







if __name__ == "__main__":
    assembling_helper("0025", "0084")
