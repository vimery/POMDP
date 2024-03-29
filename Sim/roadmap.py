from Sim.tools import *
import copy

pi = math.pi
h_pi = math.pi / 2
nh_pi = -math.pi / 2
# error of float number
epsilon = 1e-10


class Segment:
    """
    Segment: a straight road
    """

    def __init__(self, s_id, start_x, start_y, length, width, theta, max_speed=4):
        """
        Constructor, define the start point, length, angle relative to coordinate and max speed
        :param start_x: start point in x axis
        :param start_y: start point in y axis
        :param length: length of road
        :param width: width of road
        :param theta: angle relative to coordinate axis. The angle of x axis is 0
        :param max_speed: speed limits of the road
        """
        self.id = s_id
        self.x = start_x
        self.y = start_y
        self.width = width
        self.len = length
        self.theta = theta
        self.x_end = self.x + math.cos(self.theta) * self.len
        self.y_end = self.y + math.sin(self.theta) * self.len
        self.max_speed = max_speed
        self.next_seg = None

        # rect
        self._x_min = min(self.x, self.x_end) - epsilon
        self._y_min = min(self.y, self.y_end) - epsilon
        self._x_max = max(self.x, self.x_end) + epsilon
        self._y_max = max(self.y, self.y_end) + epsilon

    def next(self, x, y, theta, distance):
        return x + math.cos(self.theta) * distance, y + math.sin(self.theta) * distance, theta

    def contains(self, x, y):
        return self._x_min <= x <= self._x_max and self._y_min <= y <= self._y_max

    def get_distance_to_end(self, x, y):
        return abs(self.y_end - y) + abs(self.x_end - x)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def _get_line(self, top=-1):
        h = self.theta % pi == 0
        v = not h
        w = self.width / 2
        x, y = self.x + top * v * w, self.y + top * h * w
        x_end, y_end = self.x_end + top * v * w, self.y_end + top * h * w
        return cartesian2py(x, y), cartesian2py(x_end, y_end)

    def render(self, surface):
        start_1, end_1 = self._get_line(-1)
        start_2, end_2 = self._get_line(1)
        start_mid, end_mid = self._get_line(0)
        pg.draw.aaline(surface, Color.black, (start_1[0], start_1[1]), (end_1[0], end_1[1]))
        pg.draw.aaline(surface, Color.black, (start_2[0], start_2[1]), (end_2[0], end_2[1]))
        draw_dashed_line(surface, Color.black, start_mid, end_mid)


class Connection:
    """
    Connection: connection of two segments. Its shape is 1/4 circle or a straight line.
                     |     |     |
                     |  7  |  8  |
                     | seg | seg |
                     |     |     |
                     |     |     |
        ——————————————           —————————————————
            <---- seg2             <---- seg1
        --------------   inter   -----------------
            ----> seg3             ----> seg4
        ——————————————           —————————————————
                     |     |     |
                     | seg | seg |
                     |  5  |  6  |
                     |     |     |
                     |     |     |
                     |     |     |
    """

    def __init__(self, seg1, seg2, max_speed=4):
        """
        Constructor. Define the center, radius, left_turn of the connection
        :param seg1:
        :param seg2:
        """
        self.id = seg1.id + seg2.id
        self.x_end = seg2.x
        self.y_end = seg2.y
        # if this is a line, theta is the direction of this line
        self.theta = seg1.theta
        self.next_seg = None

        # get radius, center and direction
        if seg1.theta == seg2.theta:
            # radius, if r == 0, it's a line
            self.r = 0
            self.length = seg1.width * 2
        elif (seg1.theta < seg2.theta and (seg1.theta != nh_pi or seg2.theta != pi)) or (
                seg1.theta == pi and seg2.theta == nh_pi):
            # left turn, r is 1.5 width
            self.left_turn = 1
            self.r = seg1.width * 1.5
            self.length = h_pi * self.r
            # center of circle
            if (seg1.y_end < seg2.y) ^ (seg1.x_end < seg2.x):
                # 1st and 3rd quadrant
                self.y = seg1.y_end
                self.x = seg2.x
            else:
                self.y = seg2.y
                self.x = seg1.x_end
        else:
            # right turn, r is 0.5 width
            self.left_turn = -1
            self.r = seg1.width / 2
            self.length = h_pi * self.r
            # center of circle
            if (seg1.y_end < seg2.y) ^ (seg1.x_end < seg2.x):
                # 1st and 3rd quadrant
                self.y = seg2.y
                self.x = seg1.x_end
            else:
                self.y = seg1.y_end
                self.x = seg2.x

        # get rect
        self._x_min = min(seg1.x_end, seg2.x) - epsilon
        self._x_max = max(seg1.x_end, seg2.x) + epsilon
        self._y_min = min(seg1.y_end, seg2.y) - epsilon
        self._y_max = max(seg1.y_end, seg2.y) + epsilon

        self.max_speed = max_speed

    def contains(self, x, y):
        return self._x_min <= x <= self._x_max and self._y_min <= y <= self._y_max

    def get_distance_to_end(self, x, y):
        if self.r == 0:
            return abs(self.x_end - x) + abs(self.y_end - y)
        delta_angle = math.atan2(y - self.y, x - self.x) - self.theta
        if delta_angle > pi or delta_angle < -pi:
            delta_angle %= pi
        return self.r * abs(delta_angle)

    def get_occupancy(self):
        if self.r == 0:
            if self.theta % pi == 0:
                if self.y_end > 0:
                    return ["1", "2"]
                else:
                    return ["3", "4"]
        if self.left_turn == 1:
            if self.x > 0:
                if self.y > 0:
                    return ["1", "3", "4"]
                else:
                    return ["1", "2", "3"]
            else:
                if self.y > 0:
                    return ["2", "3", "4"]
                else:
                    return ["1", "2", "4"]
        else:
            if self.y_end > 0:
                if self.x_end > 0:
                    return ["2"]
                else:
                    return ["1"]
            else:
                if self.x_end > 0:
                    return ["4"]
                else:
                    return ["3"]

    def next(self, x, y, theta, distance):
        if self.r == 0:
            return x + math.cos(self.theta) * distance, y + math.sin(self.theta) * distance, theta
        delta_theta = self.left_turn * distance / self.r
        new_theta = delta_theta + math.atan2(y - self.y, x - self.x)
        return self.x + self.r * math.cos(new_theta), self.y + self.r * math.sin(new_theta), theta + delta_theta

    def render(self, surface):
        # do not render connection
        pass

    def get_collide_area(self, other_conn):
        cur = self.get_occupancy()
        other = other_conn.get_occupancy()
        collide = []
        for c_ele in cur:
            for o_ele in other:
                if c_ele == o_ele and c_ele not in collide:
                    collide.append(c_ele)
        return collide

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.r == other.r


class Route:
    """
    Route: a path that a car follows, contains two segments and a connection
    """

    def __init__(self, r_id, seg1, conn, seg2):
        self.id = r_id
        self.seg1 = copy.copy(seg1)
        self.conn = copy.copy(conn)
        self.seg2 = copy.copy(seg2)
        self.seg1.next_seg = self.conn
        self.conn.next_seg = self.seg2
        if conn.r == 0:
            # priority of moving direct is highest
            self.priority = 0
        elif conn.left_turn > 0:
            # left turn has higher priority
            self.priority = 1
        else:
            self.priority = 2
        self.length = seg1.len + conn.length + seg2.len

    def next(self, x, y, theta, distance):
        if self.seg2.contains(x, y):
            next_x, next_y, theta = self.seg2.next(x, y, theta, distance)
            if self.seg2.contains(next_x, next_y):
                return next_x, next_y, theta
            else:
                return None, None, None
        elif self.conn.contains(x, y):
            return self._jump_to_next_section(self.conn, self.seg2, x, y, theta, distance)
        elif self.seg1.contains(x, y):
            return self._jump_to_next_section(self.seg1, self.conn, x, y, theta, distance)

    def _jump_to_next_section(self, sec, sec2, x, y, theta, distance):
        next_x, next_y, theta = sec.next(x, y, theta, distance)
        if sec.contains(next_x, next_y):
            return next_x, next_y, theta
        else:
            return self.next(sec.x_end, sec.y_end, sec2.theta, distance - sec.get_distance_to_end(x, y))

    # def get_max_speed(self, x, y):
    #     if self.seg2.contains(x, y):
    #         return self.seg2.max_speed
    #     elif self.conn.contains(x, y):
    #         return self.conn.max_speed
    #     else:
    #         return self.seg1.max_speed

    def render(self, surface):
        self.seg1.render(surface)
        self.seg2.render(surface)
        self.conn.render(surface)

    def get_occupancy(self):
        occupancy = self.conn.get_occupancy()
        occupancy.insert(0, self.seg1.id)
        occupancy.insert(0, self.seg2.id)
        return occupancy

    def get_collide(self, other_route):
        pass

    def __eq__(self, other):
        return self.conn == other.conn


class RoadMap:
    """
    RoadMap: a static map that contains information of current environment and rules
    """

    def __init__(self, routes):
        self.routes = routes

    def get_route(self, r_id):
        return self.routes[r_id]

    def render(self, surface):
        for route in self.routes:
            route.render(surface)
