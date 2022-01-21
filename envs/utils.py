### utility functions for environments

class BulletDebugFrameDrawer(object):
    def __init__(self, sim) -> None:
        super().__init__()

        self.sim = sim

        self.axes = None

        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    def update_drawing(self, pos, quat, scale=0.1):
        if self.axes is not None:
            #remove the existing one
            for item in self.axes:
                self.sim.removeUserDebugItem(item)
        self.axes = []
        #draw new one
        rot = self.sim.getMatrixFromQuaternion(quat)
        for i in range(3):
            lineTo = [pos[j]+scale*rot[j*3+i] for j in range(3)]
            self.axes.append(self.sim.addUserDebugLine(lineFromXYZ=pos, lineToXYZ=lineTo, lineColorRGB=self.colors[i]))
        return            

