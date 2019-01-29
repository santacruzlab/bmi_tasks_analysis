from exotasks import *
from riglib import plants

from target_graphics import VirtualCircularTarget

top_view_offset = np.array([-12, 0, 0])
front_view_offset = np.array([12, 0, 0])

class VirtualActiveExoPlant(ActiveExoPlant):
    '''
    specific case of ActiveExoPlant which displays two different endpoints representing different viewing angles (top and front)
    '''
    def __init__(self, *args, **kwargs):
        super(VirtualActiveExoPlant, self).__init__(*args, **kwargs)

        self.top_view_cursor = Sphere(radius=0.4, color=arm_color) # TODO don't hardcode
        self.front_view_cursor = Sphere(radius=0.4, color=arm_color) # TODO don't hardcode

        self.graphics_models = [self.top_view_cursor, self.front_view_cursor]

    def set_intrinsic_coordinates(self, theta):
        '''
        Set the joint by specifying the angle in radians. Theta is a list of angles. If an element of theta = NaN, angle should remain the same.
        '''
        endpoint_pos = self.kin_chain.endpoint_pos(theta)

        top_view_pos = np.array([endpoint_pos[0], 0, endpoint_pos[1]]) + top_view_offset
        self.top_view_cursor.translate(*top_view_pos, reset=True)

        front_view_pos = np.array([endpoint_pos[0], 0, endpoint_pos[2]]) + front_view_offset
        self.front_view_cursor.translate(*front_view_pos, reset=True)


class VirtualMultiAngleTarget(VirtualCircularTarget):
    def _pickle_init(self):
        self.top_view_sphere = Sphere(radius=self.target_radius, color=self.target_color)
        self.front_view_sphere = Sphere(radius=self.target_radius, color=self.target_color)        
        self.graphics_models = [self.top_view_sphere, self.front_view_sphere]

        self.top_view_sphere.translate(*self.position)
        self.front_view_sphere.translate(*self.position)

    def drive_to_new_pos(self):
        self.position = self.int_position
        
        top_view_pos = np.array([self.position[0], 0, self.position[1]]) + top_view_offset
        front_view_pos = np.array([self.position[0], 0, self.position[2]]) + front_view_offset

        self.top_view_sphere.translate(*top_view_pos, reset=True)
        self.front_view_sphere.translate(*front_view_pos, reset=True)


class VirtualExoVisualFeedback(ExoBase):
    pass