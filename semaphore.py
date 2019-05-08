
from browser import document, html, svg, window

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



class Exit:
    COLORS = ['red', 'blue', 'green', 'orange', 'gold', 'brown', 'darkcyan', 'deeppink', 'olive']


def curved_path(start, end, relative=False):
    if relative:
        raise NotImplementedError
    else:
        # dcx = (end.x - start.x) / abs(end.col - start.col)
        dcx = (end.x - start.x) / 3
        return 'C {scx} {start.y} {ecx} {end.y} {end.x} {end.y}'.format(
            start=start, end=end, scx=start.x + dcx, ecx=end.x - dcx
        )

class Rail:
    ROW_TRANSFER_LENGTHS = [3, 6]
    MIN_ROW_TRANSFER_LENGTH = min(ROW_TRANSFER_LENGTHS)
    BRAKE_OFFSET = 0.5
    CRASH_OFFSET = 0.1

    # def __init__(self, node1, node2):
        # if node1.col > node2.col:
            # node1, node2 = node2, node1
        # self.left_node = node1
        # self.right_node = node2
        # self.left_node.add_right_rail(self)
        # self.right_node.add_left_rail(self)

    # def draw(self):
        # self.path = svg.
        # self.left_node.check_dead_ends()
        # self.right_node.check_dead_ends()

    @classmethod
    def path(cls, nodes):
        return 'M {0.x} {0.y}'.format(nodes[0]) + ''.join(
            # go horizontal if rows are equal
            ('H ' + str(next.x)) if prev.row == next.row
            # cubic bezier for row transfers
            else curved_path(prev, next)
            for prev, next in zip(nodes[:-1], nodes[1:])
        )


class RailSegment:
    pass



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
        # self._win_width = window.innerWidth
        # self._win_height = window.innerHeight
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
        return [self.builder_layer, self.grid_layer, self.rail_layer, self.node_layer]

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
        # print(WIDTH, HEIGHT, svg_pt.x, svg_pt.y, 
        return self.nodes[int(round(svg_pt.y / self.row_height - 1))][int(round(svg_pt.x / self.col_width - 1))]
        # return self.nodes[int(round(svg_pt.y / self.row_height))][int(round(svg_pt.x / self.col_width))]

    def row_to_y(self, row):
        return (row + 1) * self.row_height

    def col_to_x(self, col):
        return (col + 1) * self.col_width

    # def build_rail(self, node1, node2):
        # new_rail = Rail(node1, node2)
        # self.rails.append(new_rail)
        # self._update_segments(new_rail)
        # self.rail_layer <= new_rail.draw()

    def _update_segments(self, rail):
        pass

    def enable_route_highlight(self):
        pass

    def disable_route_highlight(self):
        pass

    def _generate_nodes(self):
        return [[Node(i, j) for j in range(self.cols)] for i in range(self.rows)]

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
            node.bind_drag_events(self)
        self.catcher.bind('mousemove', self.drag_moved)
        self.catcher.bind('mouseup', self.drag_ended)
        self.drag_origin = None

    def disable(self):
        self.drag_ended()
        for node in self.grid.iternodes():
            node.unbind_drag_events()
        self.catcher.unbind('mousemove')
        self.catcher.unbind('mouseup')

    def drag_started(self, event):
        self.drag_origin = self.grid.closest_node_screen(event.x, event.y)
        self.drag_origin.show_build_endpoint()
        self.drag_path = svg.path()
        self.drag_path.classList.add('build-path')
        self.layer <= self.drag_path
        self.drag_nodes = []
        self.drag_current = None

    def drag_moved(self, event):
        if not self.drag_origin:
            return
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
                self.drag_path.attrs['d'] = Rail.path(new_drag_nodes)
                tgt_opacity = 1
            else:
                tgt_opacity = 0
            self.drag_nodes = new_drag_nodes
            self.drag_path.style.stroke_opacity = tgt_opacity

    def drag_ended(self, event):
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
                current = self.grid.nodes[target.row][current.col + col_direction]
            else:
                cols_per_row = abs(current.col - target.col) / abs(current.row - target.row)
                if (cols_per_row < Rail.MIN_ROW_TRANSFER_LENGTH):
                    break # too sharp turn
                # longest row switch that is under cols_per_row
                rowswitch_length = max(
                    l for l in Rail.ROW_TRANSFER_LENGTHS if l <= cols_per_row
                )
                # move by one row and some columns in the right direction
                current = self.grid.nodes[current.row + row_direction][current.col + col_direction * rowswitch_length]
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
        print('building rail between', nodes)
        # if len(nodes) > 1:
            # for i in range(len(nodes) - 1):
                # self.grid.build_rail(nodes[i], nodes[i+1])



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

    def draw(self, x, y):
        self.x = x
        self.y = y
        self.object = svg.g(
            transform='translate({},{})'.format(x,y),
            id=self.id,
        )
        self.object.classList.add('node')
        self.object <= svg.circle(cx=0, cy=0, r=7)
        return self.object

    def get_element(self):
        return document[self.id]

    def has_rail_to(self, node):
        if node.col > self.col:
            return node in self._right_links
        else:
            return node in self._left_links

    def show_build_endpoint(self):
        self.object.classList.add('build-endpoint')

    def hide_build_endpoint(self):
        self.object.classList.remove('build-endpoint')

    def bind_drag_events(self, dragger):
        # self.object.draggable = True
        self.object.bind('mousedown', dragger.drag_started)
        self.object.bind('mousemove', dragger.drag_moved)
        self.object.bind('mouseup', dragger.drag_ended)

    def unbind_drag_events(self):
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