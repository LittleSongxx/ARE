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


def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    # assert start[0] >= map_info.map_origin_x
    # assert start[1] >= map_info.map_origin_y
    # assert end[0] >= map_info.map_origin_x
    # assert end[1] >= map_info.map_origin_y
    # assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    # assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    # assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    # assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
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


def check_collision_type(start, end, map_info):
    # Bresenham line algorithm checking with enhanced wall detection
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

    # 【增强墙体检测】使用可配置的检查半径
    check_radius = parameter.COLLISION_CHECK_RADIUS
    
    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        # 检查当前点及其邻域
        for dx_check in range(-check_radius, check_radius + 1):
            for dy_check in range(-check_radius, check_radius + 1):
                check_x = int(x) + dx_check
                check_y = int(y) + dy_check
                
                if 0 <= check_x < map.shape[1] and 0 <= check_y < map.shape[0]:
                    k = map.item(check_y, check_x)
                    if k == parameter.OCCUPIED:
                        return parameter.OCCUPIED
                    # 未知区域也视为碰撞（保守策略）
                    if k == parameter.UNKNOWN:
                        return parameter.UNKNOWN
        
        if x == x1 and y == y1:
            break
            
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    
    return parameter.FREE


def get_local_openness(coords, map_info, sample_radius=5.0, num_rays=8):
    """
    计算给定位置的局部空旷程度
    Args:
        coords: 位置坐标 (x, y)
        map_info: 地图信息
        sample_radius: 采样半径（米）
        num_rays: 射线数量
    Returns:
        average_free_distance: 平均自由距离（米），越大表示越空旷
    """
    # 如果地图信息为空，返回默认值
    if map_info is None:
        return sample_radius
    
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    free_distances = []
    
    for angle in angles:
        # 沿着每个方向投射射线
        max_steps = int(sample_radius / map_info.cell_size)
        for step in range(1, max_steps + 1):
            distance = step * map_info.cell_size
            test_x = coords[0] + distance * np.cos(angle)
            test_y = coords[1] + distance * np.sin(angle)
            test_coords = np.array([test_x, test_y])
            
            # 检查是否在地图范围内
            cell_pos = get_cell_position_from_coords(test_coords, map_info, check_negative=False)
            if (cell_pos[0] < 0 or cell_pos[0] >= map_info.map.shape[1] or
                cell_pos[1] < 0 or cell_pos[1] >= map_info.map.shape[0]):
                free_distances.append(distance)
                break
            
            # 检查是否碰到障碍物
            if map_info.map[int(cell_pos[1]), int(cell_pos[0])] == parameter.OCCUPIED:
                free_distances.append(distance)
                break
        else:
            # 没有碰到障碍物，使用最大距离
            free_distances.append(sample_radius)
    
    return np.mean(free_distances) if free_distances else sample_radius


def calculate_adaptive_resolution(coords, map_info):
    """
    根据局部环境动态计算节点分辨率
    Args:
        coords: 节点坐标
        map_info: 地图信息
    Returns:
        resolution: 自适应的节点分辨率
    """
    # 如果未启用动态分辨率或地图信息为空，返回默认值
    if not parameter.ENABLE_DYNAMIC_RESOLUTION or map_info is None:
        return parameter.NODE_RESOLUTION
    
    # 计算局部空旷程度
    openness = get_local_openness(coords, map_info)
    
    # 根据空旷程度线性插值分辨率
    # 狭窄区域（openness < NARROW_THRESHOLD）-> MIN_NODE_RESOLUTION
    # 空旷区域（openness > 2*NARROW_THRESHOLD）-> MAX_NODE_RESOLUTION
    if openness < parameter.NARROW_THRESHOLD:
        resolution = parameter.MIN_NODE_RESOLUTION
    elif openness > 2 * parameter.NARROW_THRESHOLD:
        resolution = parameter.MAX_NODE_RESOLUTION
    else:
        # 线性插值
        ratio = (openness - parameter.NARROW_THRESHOLD) / parameter.NARROW_THRESHOLD
        resolution = parameter.MIN_NODE_RESOLUTION + ratio * (parameter.MAX_NODE_RESOLUTION - parameter.MIN_NODE_RESOLUTION)
    
    return resolution


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