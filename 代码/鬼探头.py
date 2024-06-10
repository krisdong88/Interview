import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

# 初始化Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# 车的顶点（较小）
car_vertices = (
    (0.5, -0.25, -0.25),
    (0.5, 0.25, -0.25),
    (-0.5, 0.25, -0.25),
    (-0.5, -0.25, -0.25),
    (0.5, -0.25, 0.25),
    (0.5, 0.25, 0.25),
    (-0.5, -0.25, 0.25),
    (-0.5, 0.25, 0.25)
)

# 行人的顶点（更小）
pedestrian_vertices = (
    (0.2, -0.4, -0.2),
    (0.2, 0.4, -0.2),
    (-0.2, 0.4, -0.2),
    (-0.2, -0.4, -0.2),
    (0.2, -0.4, 0.2),
    (0.2, 0.4, 0.2),
    (-0.2, -0.4, 0.2),
    (-0.2, 0.4, 0.2)
)

edges = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
)

def draw_cube(vertices):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# 模拟“鬼探头”场景
def ghost_scene():
    car_position = [0, -1, 0]
    pedestrian_position = [random.uniform(-2, 2), -1, random.uniform(-2, 2)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 画车
        glPushMatrix()
        glTranslatef(*car_position)
        glColor3f(1, 0, 0)  # 车子颜色为红色
        draw_cube(car_vertices)
        glPopMatrix()

        # 画行人
        glPushMatrix()
        glTranslatef(*pedestrian_position)
        glColor3f(0, 0, 1)  # 行人颜色为蓝色
        draw_cube(pedestrian_vertices)
        glPopMatrix()

        # 更新行人位置，模拟“鬼探头”
        pedestrian_position[2] += 0.1
        if pedestrian_position[2] > 2:
            pedestrian_position = [random.uniform(-2, 2), -1, -2]

        pygame.display.flip()
        pygame.time.wait(50)

if __name__ == "__main__":
    ghost_scene()
