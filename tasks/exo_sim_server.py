'''
Server to simulate the active exo
'''

from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor
import numpy as np
import struct

Delta = 1./60

class Echo(DatagramProtocol):
    joint_angles = np.zeros(5)
    joint_velocities = np.zeros(5)
    joint_applied_torque = np.zeros(5)
    n_joints = 5

    def datagramReceived(self, data, (host, port)):
        self.joint_angles += Delta * self.joint_velocities
        if data == 's':
            pass
        else: # should be joint velocity
            vel = struct.unpack('>I' + 'd'*self.n_joints, data)
            self.joint_velocities = np.array(vel)[1:]

        vec_data = tuple(np.hstack([5, self.joint_angles, 5, self.joint_velocities, 5, self.joint_applied_torque]))
        print vec_data
        print len(vec_data)
        return_data = struct.pack('>IdddddIdddddIddddd', *vec_data)
        self.transport.write(return_data, (host, 60000))

reactor.listenUDP(60001, Echo())
reactor.run()