# Fiber information for Columns 1-6
#contains locations (x, y, z, phi) and open angles for fibers 1, 3, 5 from each column
#contains locations (x, y, z) and directions (x, y, z) or placeholders for location/direction (mostly placeholders at this point) for arrows and cones
#x, y, z are in mm, phi and open angle are in degrees
#does not contain mode or wavelength. is this necessary? unknown

'''# Mapping from col board name to the board number

columns = {""}''' #is this necessary? testing needed

# Fiber x positions
x = [
0, #placeholder/fake fiber for code testing
1005, 1005, 1005, #column 1
502.5, 502.5, 502.5, #column 2
-502.5, -502.5, -502.5, #column 3
-1005, -1005, -1005, #column 4
-502.5, -502.5, -502.5, #column 5
502.5, 502.5, 502.5 #column 6
]

# Fiber y positions
y = [
0, #placeholder/fake fiber for code testing
0, 0, 0, #column 1
870.4, 870.4, 870.4, #column 2
870.4, 870.4, 870.4, #column 3
0, 0, 0, #column 4
-870.4, -870.4, -870.4, #column 5
-870.4, -870.4, -870.4 #column 6
]

# Fiber z positions
z = [
0, #placeholder/fake fiber for code testing
571.5, 114.3, -342.9, #column 1
571.5, 114.3, -342.9, #column 2
571.5, 114.3, -342.9, #column 3
571.5, 114.3, -342.9, #column 4
571.5, 114.3, -342.9, #column 5
571.5, 114.3, -342.9 #column 6
]

#Fiber phi positions
phi = [
0, #placeholder/fake fiber for code testing
0, 0, 0, #column 1
60, 60, 60, #column 2
120, 120, 120, #column 3
180, 180, 180, #column 4
240, 240, 240, #column 5
300, 300, 300 #column 6
]

#Fiber open angles
open_angle = [
60, #placeholder/fake fiber for code testing
38, 120, 38, #column 1
38, 120, 38, #column 2
38, 120, 38, #column 3
38, 120, 38, #column 4
38, 120, 38, #column 5
38, 120, 38 #column 6
]

#cone positions
cone_positions = [
[], #placeholder/fake fiber for code testing
[], #1_1
[], #1_3
[], #1_5, have data for this one
[0, -450, 2000], #2_1
[], #2_3
[], #2_5
[], #3_1
[], #3_3
[], #3_5, have data for this one
[], #4_1
[], #4_3
[], #4_5  
[], #5_1
[], #5_3
[], #5_5
[], #6_1
[], #6_3
[] #have data for this one? labelled as 6_6. needs to be checked
]

#cone directions
cone_directions = [
[], #placeholder/fake fiber for code testing
[], #1_1
[], #1_3
[], #1_5, have data for this one
[0, 0, 1], #2_1
[], #2_3
[], #2_5
[], #3_1
[], #3_3
[], #3_5, have data for this one
[], #4_1
[], #4_3
[], #4_5  
[], #5_1
[], #5_3
[], #5_5
[], #6_1
[], #6_3
[] #have data for this one? labelled as 6_6. needs to be checked
]

#arrow positions
arrow_positions = [
[], #placeholder/fake fiber for code testing
[1005, 0, 571.5], #1_1
[1005, 0, 114.3], #1_3
[1005, 0, -342.9], #1_5, have data for this one
[502.0, 870.4, 571.5], #2_1
[502.0, 870.4, 114.3], #2_3
[502.0, 870.4, -342.9], #2_5
[-502.5, 870.4, 571.5], #3_1
[-502.5, 870.4, 114.3], #3_3
[-502.5, 870.4, -342.9], #3_5, have data for this one
[-1005, 0, 571.5], #4_1
[-1005, 0, 114.3], #4_3
[-1005, 0, -342.9], #4_5  
[-502.5, -870.4, 571.5], #5_1
[-502.5, -870.4, 114.3], #5_3
[-502.5, -870.4, -342.9], #5_5
[502.5, -870.4, 571.5], #6_1
[502.5, -870.4, 114.3], #6_3
[502.5, -870.4, -342.9] #6_5. have data for this one? labelled as 6_6. needs to be checked
]

#arrow directions
[], #placeholder/fake fiber for code testing
[], #1_1
[], #1_3
[], #1_5, have data for this one
[0, 1, 0], #2_1
[], #2_3
[], #2_5
[], #3_1
[], #3_3
[], #3_5, have data for this one
[], #4_1
[], #4_3
[], #4_5  
[], #5_1
[], #5_3
[], #5_5
[], #6_1
[], #6_3
[] #have data for this one? labelled as 6_6. needs to be checked
]
