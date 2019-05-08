
from browser import document, html, svg

WIDTH = 1200
HEIGHT = 530


class Train:
    pass


class Switch:
    MARKER = svg.marker(
        id='switcharrow',
        markerWidth=3, markerHeight=2,
        refX=2, refY=1,
        orient='auto',
        markerUnits='strokeWidth',
    )
    MARKER <= svg.path(d='M0,0 L0,2 L3,1 L0,0')
    enabled = False
    LENGTH_FRACTION = 0.4

    def __init__(self, node, left):
        self.node = node
        self.left = left
        self.dirtext = 'left' if self.left else 'right'
        self.position = 0
        self.arrow = None

    def draw(self):
        d = self._path()
        self.arrow = svg.path(d=d)
        self.arrow.bind('click', self.switch)
        self.arrow.classList.add('switch')
        self.arrow.classList.add('switch-' + self.dirtext)
        return self.arrow

    def _path(self):
        target = self.node.get_link(self.left, self.position)
        part = self.LENGTH_FRACTION / (abs(target.col - self.node.col) ** .75)
        return rail_segment_d(self.node, target, part, relative=True)

    def redraw(self):
        if self.arrow is None:
            return self.draw()

    def switch(self, event=None):
        if self.enabled:
            print('switch ', self, ' flipped')

    def __repr__(self):
        return '<{}Switch{}({})>'.format(
            self.dirtext.capitalize(), self.node, self.position
        )


class Exit:
    COLORS = [
        'red', 'blue', 'green',
        'orange', 'gold', 'brown', 'darkcyan', 'deeppink', 'olive'
    ]


def curve_coors(start, end, part=1):
    dx = end.x - start.x
    dy = end.y - start.y
    dcx = dx / 3
    # dcx = (end.x - start.x) / abs(end.col - start.col)
    if dy == 0:
        return (dx * part,)
    else:
        coors = (dcx, 0, dx - dcx, dy, dx, dy)
        if part == 1:
            return coors
        else:
            return curve_split(coors, part)


def curve_split(coors, part=1):
    # performs de Casteljau algorithm for bezier curve splitting
    # coors: control points as [x0, y0, x1, y1, x2, y2...]
    # t in [0,1] is the percent where the curve is split
    degree = len(coors) / 2  # should be 3 for cubic curves
    current = (0, 0) + coors
    coefs = [current]
    while len(coefs) <= degree:
        previous = current
        current = []
        for i in range(0, len(previous) - 2, 2):
            current.append(previous[i] * (1-part) + previous[i+2] * part)
            current.append(previous[i+1] * (1-part) + previous[i+3] * part)
        coefs.append(current)
    # return []
    result = tuple(val for seg in coefs[1:] for val in seg[:2])
    return result


CURVE_SEGMENT_PATTERN = ' '.join(['{}'] * 6)


def rail_path_d(nodes):
    ds = []
    for prev, next in zip(nodes[:-1], nodes[1:]):
        coors = curve_coors(prev, next)
        if len(coors) == 1:
            ds.append('h' + str(coors[0]))
        else:
            ds.append('c' + CURVE_SEGMENT_PATTERN.format(*coors))
    return 'M{0.x} {0.y}'.format(nodes[0]) + ''.join(ds)


def rail_segment_d(start, end, part=1, relative=False):
    coors = curve_coors(start, end, part=part)
    return (
        'M0 0' if relative else 'M{0.x} {0.y}'.format(start)
    ) + (
        ('h' + str(coors[0])) if len(coors) == 1
        else 'c' + CURVE_SEGMENT_PATTERN.format(*coors)
    )


class Rail:
    ROW_TRANSFER_LENGTHS = [3, 6]
    MIN_ROW_TRANSFER_LENGTH = min(ROW_TRANSFER_LENGTHS)
    BRAKE_OFFSET = 0.5
    CRASH_OFFSET = 0.1

    def __init__(self, node1, node2):
        if node1.col > node2.col:
            node1, node2 = node2, node1
        self.left_node = node1
        self.right_node = node2
        self.left_node.add_right_rail(self)
        self.right_node.add_left_rail(self)
        self.crossings = []
        self.path = None

    def draw(self):
        self.path = svg.path(
            d=rail_segment_d(self.left_node, self.right_node)
        )
        self.path.classList.add('rail')
        if self.crossings:
            self.path.classList.add('crossed')
        self.left_node.redraw()
        self.right_node.redraw()
        return self.path

    def get_crossings(self, rails):
        if self.left_node.row == self.right_node.row:
            return []
        row_splits = {
            self.left_node.row: self.left_node.col,
            self.right_node.row: self.right_node.col,
        }
        for rail in rails:
            crossable = (
                rail.left_node.row != rail.right_node.row
                and rail.left_node.row in row_splits
                and rail.right_node.row in row_splits
            )
            if crossable:
                left_split = row_splits[rail.left_node.row]
                right_split = row_splits[rail.right_node.row]
                crosses = (
                    rail.left_node.col < left_split
                    and rail.right_node.col > right_split
                ) or (
                    rail.left_node.col > left_split
                    and rail.right_node.col < right_split
                )
                if crosses:
                    yield rail

    def add_crossing(self, rail):
        ml = self.left_node
        rl = rail.left_node
        mr = self.right_node
        rr = rail.right_node
        # compute line intersection (approximate)
        t = (
            ml.col * (rl.row - rr.row)
            + rl.col * (rr.row - ml.row)
            + rr.col * (ml.row - rl.row)
        ) / (
            (ml.col - mr.col) * (rl.row - rr.row)
            + (rl.col - rr.col) * (mr.row - ml.row)
        )
        if self.path:
            self.path.classList.add('crossed')
        self.crossings.append((rail, t))

    def __repr__(self):
        return '<Rail{0.left_node}-{0.right_node}>'.format(self)


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.nodes = self._generate_nodes()
        self.rails = []
        self.segments = []
        self.switches = []
        self.exits = []
        self.builder = RailBuilder(self)
        self.resize()

    def iternodes(self):
        for noderow in self.nodes:
            for node in noderow:
                yield node

    def resize(self):
        self._win_width = WIDTH
        self._win_height = HEIGHT
        self.row_height = self._win_height / (self.rows + 1)
        self.col_width = self._win_width / (self.cols + 1)
        self.row_ys = [(row + 1) * self.row_height for row in range(self.rows)]
        self.col_xs = [(col + 1) * self.col_width for col in range(self.cols)]

    def draw(self, svg_interface):
        self.svg_interface = svg_interface
        self.builder_layer = self.builder.draw()
        self.grid_layer = self.draw_grid()
        self.rail_layer = svg.g(id='raillayer')
        self.node_layer = self.draw_nodes()
        self.aux_pt = self.svg_interface.createSVGPoint()
        return [
            self.builder_layer,
            self.grid_layer,
            self.rail_layer,
            self.node_layer,
        ]

    def draw_grid(self):
        layer = svg.g(id='gridlayer')
        for y in self.row_ys:
            line = svg.line(x1=0, x2=self._win_width, y1=y, y2=y)
            line.classList.add('grid')
            layer <= line
        for x in self.col_xs:
            line = svg.line(x1=x, x2=x, y1=0, y2=self._win_height)
            line.classList.add('grid')
            layer <= line
        return layer

    def draw_nodes(self):
        layer = svg.g(id='nodelayer')
        for y, row_nodes in zip(self.row_ys, self.nodes):
            for x, node in zip(self.col_xs, row_nodes):
                layer <= node.draw(x, y)
        return layer

    def closest_node_screen(self, x_scr, y_scr):
        self.aux_pt.x = x_scr
        self.aux_pt.y = y_scr
        transform = self.svg_interface.getScreenCTM().inverse()
        svg_pt = self.aux_pt.matrixTransform(transform)
        return self.nodes[
            int(round(svg_pt.y / self.row_height - 1))
        ][
            int(round(svg_pt.x / self.col_width - 1))
        ]

    def row_to_y(self, row):
        return (row + 1) * self.row_height

    def col_to_x(self, col):
        return (col + 1) * self.col_width

    def build_rail(self, node1, node2):
        if not node1.has_rail_to(node2):
            new_rail = Rail(node1, node2)
            print('building', new_rail)
            for rail in new_rail.get_crossings(self.rails):
                rail.add_crossing(new_rail)
                new_rail.add_crossing(rail)
            self.rails.append(new_rail)
            self.rail_layer <= new_rail.draw()

    def _update_segments(self, rail):
        pass

    def enable_route_highlight(self):
        pass

    def disable_route_highlight(self):
        pass

    def _generate_nodes(self):
        return [
            [Node(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ]

    def __repr__(self):
        return '<Grid({0.rows}x{0.cols})>'.format(self)


class RailBuilder:
    def __init__(self, grid):
        self.grid = grid

    def draw(self):
        self.layer = svg.g(id='buildlayer')
        self.catcher = svg.rect(
            x=0, y=0, width=WIDTH, height=HEIGHT,
            style={'fill': '#fff'}
        )
        self.layer <= self.catcher
        return self.layer

    def enable(self):
        for node in self.grid.iternodes():
            node.enable_building(self)
        self.catcher.bind('mousemove', self.drag_moved)
        self.catcher.bind('mouseup', self.drag_ended)
        self.drag_path = None
        self.drag_origin = None

    def disable(self):
        self.drag_ended()
        for node in self.grid.iternodes():
            node.disable_building()
        self.catcher.unbind('mousemove')
        self.catcher.unbind('mouseup')

    def drag_started(self, event):
        if self.drag_origin:
            return
        print('drag started')
        self.drag_origin = self.grid.closest_node_screen(event.x, event.y)
        self.drag_origin.show_build_endpoint()
        if self.drag_path:
            self.drag_path.remove()
        self.drag_path = svg.path()
        self.drag_path.classList.add('build-path')
        self.layer <= self.drag_path
        self.drag_nodes = []
        self.drag_current = None

    def drag_moved(self, event):
        if not self.drag_origin:
            return
        print('drag moved')
        drag_current = self.grid.closest_node_screen(event.x, event.y)
        if drag_current != self.drag_current:
            self.drag_current = drag_current
            new_drag_nodes = self.shortest_unbuilt_path(
                self.drag_origin, self.drag_current
            )
            if self.drag_nodes:
                self.drag_nodes[-1].hide_build_endpoint()
            if new_drag_nodes:
                new_drag_nodes[-1].show_build_endpoint()
                self.drag_path.attrs['d'] = rail_path_d(new_drag_nodes)
                tgt_opacity = 1
            else:
                tgt_opacity = 0
            self.drag_nodes = new_drag_nodes
            self.drag_path.style.stroke_opacity = tgt_opacity

    def drag_ended(self, event=None):
        print('drag ended')
        if not self.drag_origin:
            return
        self.drag_origin.hide_build_endpoint()
        self.drag_origin = None
        self.drag_path.remove()
        if self.drag_nodes:
            self.drag_nodes[-1].hide_build_endpoint()
            self.build_rails(self.drag_nodes)

    def shortest_unbuilt_path(self, origin, target):
        path = [origin]
        current = origin
        row_direction = 1 if origin.row < target.row else -1
        col_direction = 1 if origin.col < target.col else -1
        while current.col != target.col:
            if current.row == target.row:
                # straight path
                next_col = current.col + col_direction
                current = self.grid.nodes[target.row][next_col]
            else:
                cols_per_row = (
                    abs(current.col - target.col)
                    / abs(current.row - target.row)
                )
                if (cols_per_row < Rail.MIN_ROW_TRANSFER_LENGTH):
                    break  # too sharp turn
                # longest row switch that is under cols_per_row
                rowswitch_length = max(
                    l for l in Rail.ROW_TRANSFER_LENGTHS if l <= cols_per_row
                )
                # move by one row and some columns in the right direction
                current = self.grid.nodes[
                    current.row + row_direction
                ][
                    current.col + col_direction * rowswitch_length
                ]
            path.append(current)
        return self.strip_built_parts(path)

    @staticmethod
    def strip_built_parts(path):
        if len(path) >= 2:
            # strip ends that are already built
            start_i = 0
            while path[start_i].has_rail_to(path[start_i+1]):
                start_i += 1
                if start_i == len(path) - 1:
                    break
            path = path[start_i:]
            if len(path) >= 2:
                end_i = len(path) - 1
                while path[end_i].has_rail_to(path[end_i-1]):
                    end_i -= 1
                    if end_i == 0:
                        break
                path = path[:end_i+1]
        return path if len(path) >= 2 else []

    def build_rails(self, nodes):
        if len(nodes) > 1:
            for i in range(len(nodes) - 1):
                self.grid.build_rail(nodes[i], nodes[i+1])


class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = None
        self.y = None
        self.id = 'n{0.row}-{0.col}'.format(self)
        self.left_rails = []
        self.right_rails = []
        self._left_links = []
        self._right_links = []
        self.left_switch = None
        self.right_switch = None

    def get_link(self, left, index):
        return (self._left_links if left else self._right_links)[index]

    def add_left_rail(self, rail):
        self.left_rails.append(rail)
        self._left_links.append(rail.left_node)
        if len(self.left_rails) == 2:
            self.left_switch = Switch(self, True)

    def add_right_rail(self, rail):
        self.right_rails.append(rail)
        self._right_links.append(rail.right_node)
        if len(self.right_rails) == 2:
            self.right_switch = Switch(self, False)

    def draw(self, x, y):
        self.x = x
        self.y = y
        self.object = svg.g(
            transform='translate({},{})'.format(x, y),
            id=self.id,
        )
        self.object.classList.add('node')
        self.object <= svg.circle(cx=0, cy=0, r=7)
        self.dead_end_stop = None
        return self.object

    def get_element(self):
        return document[self.id]

    def has_rail_to(self, node):
        if node.col > self.col:
            return node in self._right_links
        else:
            return node in self._left_links

    def redraw(self):
        self.redraw_dead_end()
        self.redraw_switch(self.left_switch)
        self.redraw_switch(self.right_switch)

    def redraw_switch(self, sw):
        if sw is not None:
            switch_object = sw.redraw()
            if switch_object:
                self.object <= switch_object

    def redraw_dead_end(self):
        is_dead_end = (len(self.left_rails) == 0 or len(self.right_rails) == 0)
        if is_dead_end and not self.dead_end_stop:
            self.dead_end_stop = svg.rect(
                x=-2, y=-5, width=4, height=10,
                id='deadend-' + self.id,
            )
            self.dead_end_stop.classList.add('deadend')
            self.object <= self.dead_end_stop
        elif not is_dead_end and self.dead_end_stop:
            self.dead_end_stop.remove()
            self.dead_end_stop = None

    def show_build_endpoint(self):
        self.object.classList.add('build-endpoint')

    def hide_build_endpoint(self):
        self.object.classList.remove('build-endpoint')

    def enable_building(self, dragger):
        self.object.classList.add('building')
        self.object.bind('mousedown', dragger.drag_started)
        self.object.bind('mousemove', dragger.drag_moved)
        self.object.bind('mouseup', dragger.drag_ended)

    def disable_building(self):
        self.object.classList.remove('building')
        self.object.unbind('mousedown')
        self.object.unbind('mousemove')
        self.object.unbind('mouseup')

    def __repr__(self):
        return '<Node({0.row},{0.col})>'.format(self)


class TrainController:
    def __init__(self):
        self.trains = []


class Scheduler:
    def __init__(self, controller=None):
        self.controller = controller

    def set_controller(self, controller):
        self.controller = controller


class ModeController:
    TEXTS = {
        True: 'To route mode',
        False: 'To build mode',
    }

    def __init__(self, grid):
        self.grid = grid
        self.active = True

    def draw(self):
        self.button = html.BUTTON(self.TEXTS[self.active])
        self.button.bind('click', self.flip_active)
        document['body'] <= self.button

    def flip_active(self, event):
        if self.active:
            self.deactivate()
        else:
            self.activate()
        self.button.innerHTML = self.TEXTS[self.active]

    def activate(self):
        self.grid.disable_route_highlight()
        Switch.enabled = False
        self.grid.builder.enable()
        self.active = True

    def deactivate(self):
        self.grid.enable_route_highlight()
        Switch.enabled = True
        self.grid.builder.disable()
        self.active = False


class Game:
    SVG_ATTRS = {
        'preserveAspectRatio': 'xMinYMin meet',
        'viewBox': '0 0 {} {}'.format(WIDTH, HEIGHT),
        'class': 'svg-content-responsive'
    }

    def __init__(self, grid, scheduler):
        self.grid = grid
        self.scheduler = scheduler
        self.mode_controller = ModeController(self.grid)
        self.train_controller = TrainController()
        self.scheduler.set_controller(self.train_controller)

    def draw(self):
        self.mode_controller.draw()
        container = html.DIV()
        container.classList.add('svg-container')
        self.svg = self._create_svg()
        container <= self.svg
        document['body'] <= container
        for g in self.grid.draw(self.svg):
            self.svg <= g

    def _create_svg(self):
        svg_el = html.SVG(**self.SVG_ATTRS)
        defs = svg.defs()
        defs <= Switch.MARKER
        svg_el <= defs
        return svg_el

    def start(self):
        self.mode_controller.activate()


game = Game(Grid(15, 30), Scheduler())
game.draw()
game.start()
