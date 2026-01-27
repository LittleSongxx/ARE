import numpy as np
from skimage.morphology import label
import quads
import parameter

def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords,
                                                                                            map_info.map_origin_x,
                                                                                            map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == parameter.FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_quad_tree_box(coords, box_size):
    min_x = coords[0] - box_size / 2
    min_y = coords[1] - box_size / 2
    max_x = coords[0] + box_size / 2
    max_y = coords[1] + box_size / 2
    min_x = np.round(min_x, 1)
    min_y = np.round(min_y, 1)
    max_x = np.round(max_x, 1)
    max_y = np.round(max_y, 1)

    neighbor_boundary = quads.BoundingBox(min_x, min_y, max_x, max_y)
    return neighbor_boundary


def get_free_and_connected_map(location, map_info):
    # a binary map for free and connected areas
    free = (map_info.map == parameter.FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


def get_free_and_connected_map_extended(location, map_info):
    """
    扩展版本：将FREE和UNKNOWN都视为可能可通行的区域
    用于狭窄通道检测场景
    """
    # FREE和UNKNOWN都视为潜在可通行
    passable = ((map_info.map == parameter.FREE) | (map_info.map == parameter.UNKNOWN)).astype(float)
    labeled_passable = label(passable, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_passable[cell[1], cell[0]]
    connected_passable_map = (labeled_passable == label_number)
    return connected_passable_map


def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * parameter.CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * parameter.CELL_SIZE

    if x_min % parameter.NODE_RESOLUTION != 0:
        x_min = (x_min // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION
    if x_max % parameter.NODE_RESOLUTION != 0:
        x_max = x_max // parameter.NODE_RESOLUTION * parameter.NODE_RESOLUTION
    if y_min % parameter.NODE_RESOLUTION != 0:
        y_min = (y_min // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION
    if y_max % parameter.NODE_RESOLUTION != 0:
        y_max = y_max // parameter.NODE_RESOLUTION * parameter.NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, parameter.NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, parameter.NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == parameter.FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map


def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]

    unknown = (map_info.map == parameter.UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == parameter.FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and parameter.FRONTIER_CELL_SIZE != parameter.CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1, 2)
        frontier_coords = frontier_down_sample(frontier_coords, parameter.FRONTIER_CELL_SIZE)
    else:
        frontier_coords = set(map(tuple, frontier_coords))

    return frontier_coords


def frontier_down_sample(data, voxel_size):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data


def is_free(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != parameter.FREE:
        return False
    else:
        return True


def is_passable(location, map_info):
    """
    检查位置是否可通行（FREE或UNKNOWN都视为可通行）
    用于狭窄通道场景
    """
    cell = get_cell_position_from_coords(location, map_info)
    cell_value = map_info.map[cell[1], cell[0]]
    if cell_value == parameter.FREE or cell_value == parameter.UNKNOWN:
        return True
    return False


def check_collision(start, end, map_info):
    """
    碰撞检测函数 - 支持多种模式
    根据parameter中的配置选择不同的检测策略
    """
    # 根据配置选择检测模式
    if parameter.SOFT_COLLISION_CHECK:
        return check_collision_soft(start, end, map_info)
    elif parameter.ALLOW_UNKNOWN_PASSTHROUGH:
        return check_collision_with_unknown_tolerance(start, end, map_info)
    else:
        return check_collision_original(start, end, map_info)


def check_collision_original(start, end, map_info):
    """
    原始碰撞检测函数 - Bresenham line algorithm
    OCCUPIED和UNKNOWN都视为碰撞
    """
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == parameter.OCCUPIED:
            collision = True
            break
        if k == parameter.UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision


def check_collision_soft(start, end, map_info):
    """
    软碰撞检测 - 只有OCCUPIED才算碰撞
    UNKNOWN区域可以通过
    """
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == parameter.OCCUPIED:
            collision = True
            break
        # UNKNOWN不再视为碰撞
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision


def check_collision_with_unknown_tolerance(start, end, map_info):
    """
    带未知区域容忍度的碰撞检测
    允许穿过一定数量的连续UNKNOWN格子
    """
    collision = False
    unknown_count = 0
    max_unknown = parameter.UNKNOWN_TOLERANCE_CELLS

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == parameter.OCCUPIED:
            collision = True
            break
        if k == parameter.UNKNOWN:
            unknown_count += 1
            if unknown_count > max_unknown:
                collision = True
                break
        else:
            # 遇到FREE，重置计数
            unknown_count = 0
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision


def check_collision_type(start, end, map_info):
    # Bresenham line algorithm checking
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map.astype(np.int32)

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == parameter.OCCUPIED:
            return parameter.OCCUPIED
        if k == parameter.UNKNOWN:
            # 根据配置决定是否将UNKNOWN视为碰撞
            if not parameter.SOFT_COLLISION_CHECK:
                return parameter.UNKNOWN
            # 当SOFT_COLLISION_CHECK=True时，UNKNOWN不视为碰撞，继续检测
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return parameter.FREE


def check_collision_type_soft(start, end, map_info):
    """
    软碰撞类型检测 - 只返回OCCUPIED或FREE
    """
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map.astype(np.int32)

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == parameter.OCCUPIED:
            return parameter.OCCUPIED
        # UNKNOWN不再视为碰撞
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return parameter.FREE


def detect_narrow_passage(location, map_info, search_radius=None):
    """
    检测当前位置附近是否存在狭窄通道
    返回: (is_narrow, passage_direction) 
    """
    if not parameter.ENABLE_NARROW_PASSAGE_DETECTION:
        return False, None
    
    if search_radius is None:
        search_radius = parameter.SENSOR_RANGE / 2
    
    cell = get_cell_position_from_coords(location, map_info)
    cell_radius = int(search_radius / map_info.cell_size)
    
    # 检查8个方向
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    
    narrow_directions = []
    threshold_cells = int(parameter.NARROW_PASSAGE_WIDTH_THRESHOLD / map_info.cell_size)
    
    for dx, dy in directions:
        # 沿方向检测通道宽度
        width = 0
        for dist in range(1, cell_radius):
            check_x = cell[0] + dx * dist
            check_y = cell[1] + dy * dist
            
            if 0 <= check_x < map_info.map.shape[1] and 0 <= check_y < map_info.map.shape[0]:
                if map_info.map[check_y, check_x] == parameter.FREE:
                    # 检测垂直于行进方向的宽度
                    perp_width = measure_perpendicular_width(
                        (check_x, check_y), (dx, dy), map_info
                    )
                    if perp_width < threshold_cells:
                        narrow_directions.append((dx, dy))
                        break
    
    if narrow_directions:
        return True, narrow_directions
    return False, None


def measure_perpendicular_width(cell, direction, map_info):
    """
    测量垂直于给定方向的通道宽度
    """
    dx, dy = direction
    # 垂直方向
    perp_dx, perp_dy = -dy, dx
    
    width = 1  # 当前格子
    
    # 正向检测
    for i in range(1, 20):
        check_x = cell[0] + perp_dx * i
        check_y = cell[1] + perp_dy * i
        if 0 <= check_x < map_info.map.shape[1] and 0 <= check_y < map_info.map.shape[0]:
            if map_info.map[check_y, check_x] == parameter.FREE:
                width += 1
            else:
                break
        else:
            break
    
    # 反向检测
    for i in range(1, 20):
        check_x = cell[0] - perp_dx * i
        check_y = cell[1] - perp_dy * i
        if 0 <= check_x < map_info.map.shape[1] and 0 <= check_y < map_info.map.shape[0]:
            if map_info.map[check_y, check_x] == parameter.FREE:
                width += 1
            else:
                break
        else:
            break
    
    return width


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
