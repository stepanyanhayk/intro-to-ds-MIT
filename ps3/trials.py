width = 3
height = 1
dirt_amount = 4

tiles = {}

for i in range(width):
    for j in range(height):
        tiles[(i, j)] = dirt_amount

for tile in tiles.keys():
    print(tile[0])
    print(tile[1])
