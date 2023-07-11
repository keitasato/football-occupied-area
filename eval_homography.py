import pickle
import numpy as np

with open('coords.pickle', 'rb') as f:
    coords = pickle.load(f)
with open('coords_dlt.pickle', 'rb') as g:
    coords_dlt = pickle.load(g)

#coords = open("coords.pickle", 'rb')
#coords_dlt = open("coords_dlt.pickle", 'rb')
#coords = pickle.load(coords)
#coords_dlt = pickle.load(coords_dlt)
#coords.close()
#coords_dlt.close()

frame_index = 0
sum_error = 0
for c, d in zip(coords, coords_dlt):
    coord = np.array(c)
    coord = coord.reshape((coord.shape[1], coord.shape[2]))
    coord_dlt = np.array(d)
    coord_dlt = coord_dlt.reshape((coord_dlt.shape[1], coord_dlt.shape[2]))

    num_c = coord.shape[0]
    num_d = coord_dlt.shape[0]
    min = num_c
    if num_d < num_c:
        min = num_d
    print("frame index = ", frame_index)
    frame_index += 1
    print(coord.shape)
    print(coord_dlt.shape)
    sum_error += abs(np.sum(coord[:min, :] - coord_dlt[:min, :]))

print("Error Sum = ", sum_error)
print("Mean Error Sum = ", sum_error / frame_index)



