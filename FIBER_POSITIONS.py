# Fiber information for Columns 1-6
#contains locations (x, y, z, phi) and open angles for fibers 1, 3, 5 from each column
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
90, #placeholder/fake fiber for code testing
38, 120, 38, #column 1
38, 120, 38, #column 2
38, 120, 38, #column 3
38, 120, 38, #column 4
38, 120, 38, #column 5
38, 120, 38 #column 6
]
