# -*- coding: utf-8 -*-

import pygame
import numpy as np
#import math
#import matplotlib.pyplot as plt
#from HaplyHAPI import Board, Device, Mechanisms, Pantograph
#import sys, serial, glob
#from serial.tools import list_ports
import time

class Graphics:
    def __init__(self,device_connected,window_size=(600,400)):
        self.device_connected = device_connected
        
        #initialize pygame window
        self.window_size = window_size #default (600,400)
        pygame.init()
        self.window = pygame.display.set_mode((window_size[0]*2, window_size[1]))   ##twice 600x400 for haptic and VR
        pygame.display.set_caption('Virtual Haptic Device')

        self.screenHaptics = pygame.Surface(self.window_size)
        self.screenVR = pygame.Surface(self.window_size)

        ##add nice icon from https://www.flaticon.com/authors/vectors-market
        self.icon = pygame.image.load('robot.png')
        pygame.display.set_icon(self.icon)

        ##add text on top to debugToggle the timing and forces
        self.font = pygame.font.Font('freesansbold.ttf', 18)

        pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' toggles it
         
        ##set up the on-screen debugToggle
        self.text = self.font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (10, 10)

        #xc,yc = screenVR.get_rect().center ##center of the screen

        ##initialize "real-time" clock
        self.clock = pygame.time.Clock()
        self.FPS = 100   #in Hertz

        ##define some colors
        self.cWhite = (255,255,255)
        self.cDarkblue = (36,90,190)
        self.cLightblue = (0,176,240)
        self.cRed = (255,0,0)
        self.cOrange = (255,100,0)
        self.cYellow = (255,255,0)
        
        self.hhandle = pygame.image.load('handle.png') #
        
        self.haptic_width = 48
        self.haptic_height = 48
        self.haptic  = pygame.Rect(*self.screenHaptics.get_rect().center, 0, 0).inflate(self.haptic_width, self.haptic_height)
        self.effort_cursor  = pygame.Rect(*self.haptic.center, 0, 0).inflate(self.haptic_width, self.haptic_height) 
        self.colorHaptic = self.cOrange ##color of the wall

        ####Pseudo-haptics dynamic parameters, k/b needs to be <1
        self.sim_k = 0.5 #0.1#0.5       ##Stiffness between cursor and haptic display
        self.sim_b = 0.8 #1.5#0.8       ##Viscous of the pseudohaptic display
        
        self.window_scale = 5000 #pixels per meter
        self.device_origin = (int(self.window_size[0]/2.0 + 0.038/2.0*self.window_scale),0)
        
        self.show_linkages = True
        self.show_debug = True
        self.hide_haptic_vr = False  # set True to suppress haptic square on screenVR

    def convert_pos(self,*positions):
        #invert x because of screen axes
        # 0---> +X
        # |
        # |
        # v +Y
        converted_positions = []
        for physics_pos in positions:
            x = self.device_origin[0]-physics_pos[0]*self.window_scale
            y = self.device_origin[1]+physics_pos[1]*self.window_scale
            converted_positions.append([x,y])
        if len(converted_positions)<=0:
            return None
        elif len(converted_positions)==1:
            return converted_positions[0]
        else:
            return converted_positions
        return [x,y]
    def inv_convert_pos(self,*positions):
        #convert screen positions back into physical positions
        converted_positions = []
        for screen_pos in positions:
            x = (self.device_origin[0]-screen_pos[0])/self.window_scale
            y = (screen_pos[1]-self.device_origin[1])/self.window_scale
            converted_positions.append([x,y])
        if len(converted_positions)<=0:
            return None
        elif len(converted_positions)==1:
            return converted_positions[0]
        else:
            return converted_positions
        return [x,y]
        
    def get_events(self):
        #########Process events  (Mouse, Keyboard etc...)#########
        events = pygame.event.get()
        keyups = []
        for event in events:
            if event.type == pygame.QUIT: #close window button was pressed
                sys.exit(0) #raises a system exit exception so any Finally will actually execute
            elif event.type == pygame.KEYUP:
                keyups.append(event.key)
        
        mouse_pos = pygame.mouse.get_pos()
        return keyups, mouse_pos

    def sim_forces(self,pE,f,pM,mouse_k=None,mouse_b=None):
        #simulated device calculations
        if mouse_k is not None:
            self.sim_k = mouse_k
        if mouse_b is not None:
            self.sim_b = mouse_b
        if not self.device_connected:
            pP = self.haptic.center
            #pM is where the mouse is
            #pE is where the position is pulled towards with the spring and damping factors
            #pP is where the actual haptic position ends up as
            diff = np.array(( pM[0]-pE[0],pM[1]-pE[1]) )
            #diff = np.array(( pM[0]-pP[0],pM[1]-pP[1]) )
            
            scale = self.window_scale/1e3
            scaled_vel_from_force = np.array(f)*scale/self.sim_b
            vel_from_mouse_spring = (self.sim_k/self.sim_b)*diff
            dpE = vel_from_mouse_spring - scaled_vel_from_force
            #dpE = -dpE
            #if diff[0]!=0:
            #    if (diff[0]+dpE[0])/diff[0]<0:
            #        #adding dpE has changed the sign (meaning the distance that will be moved is greater than the original displacement
            #        #prevent the instantaneous velocity from exceeding the original displacement (doesn't make physical sense)
            #        #basically if the force given is so high that in a single "tick" it would cause the endpoint to move back past it's original position...
            #        #whatever thing is exerting the force should basically be considered a rigid object
            #        dpE[0] = -diff[0]
            #if diff[1]!=1:
            #    if (diff[1]+dpE[1])/diff[1]<0:
            #        dpE[1] = -diff[1]
            if abs(dpE[0])<1:
                dpE[0] = 0
            if abs(dpE[1])<1:
                dpE[1] = 0
            pE = np.round(pE+dpE) #update new positon of the end effector
            
            #Change color based on effort
            cg = 255-np.clip(np.linalg.norm(self.sim_k*diff/self.window_scale)*255*20,0,255)
            cb = 255-np.clip(np.linalg.norm(self.sim_k*diff/self.window_scale)*255*20,0,255)
            self.effort_color = (255,cg,cb)
        return pE

    def erase_screen(self):
        self.screenHaptics.fill(self.cWhite) #erase the haptics surface
        self.screenVR.fill(self.cLightblue) #erase the VR surface
        self.debug_text = ""
    
    def render(self,pA0,pB0,pA,pB,pE,f,pM):
        ###################Render the Haptic Surface###################
        #set new position of items indicating the endpoint location
        self.haptic.center = pE #the hhandle image and effort square will also use this position for drawing
        self.effort_cursor.center = self.haptic.center

        if self.device_connected:
            self.effort_color = (255,255,255)

        #pygame.draw.rect(self.screenHaptics, self.effort_color, self.haptic,border_radius=4)
        pygame.draw.rect(self.screenHaptics, self.effort_color, self.effort_cursor,border_radius=8)

        ######### Robot visualization ###################
        if self.show_linkages:
            pantographColor = (150,150,150)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pA0,pA],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pB0,pB],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pA,pE],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pB,pE],15)
            
            for p in ( pA0,pB0,pA,pB,pE):
                pygame.draw.circle(self.screenHaptics, (0, 0, 0),p, 15)
                pygame.draw.circle(self.screenHaptics, (200, 200, 200),p, 6)
        
        ### Hand visualisation
        self.screenHaptics.blit(self.hhandle,self.effort_cursor)
        
        #pygame.draw.line(self.screenHaptics, (0, 0, 0), (self.haptic.center),(self.haptic.center+2*k*(xm-xh)))
        
        ###################Render the VR surface###################
        if not self.hide_haptic_vr:
            pygame.draw.rect(self.screenVR, self.colorHaptic, self.haptic, border_radius=8)
        
        if not self.device_connected:
            pygame.draw.lines(self.screenHaptics, (0,0,0), False,[self.effort_cursor.center,pM],2)
        ##Fuse it back together
        self.window.blit(self.screenHaptics, (0,0))
        self.window.blit(self.screenVR, (self.window_size[0],0))

        ##Print status in  overlay
        if self.show_debug:    
            self.debug_text += "FPS = " + str(round(self.clock.get_fps()))+" "
            self.debug_text += "fe: "+str(np.round(f[0],1))+","+str(np.round(f[1],1))+"] "
            self.debug_text += "xh: ["+str(np.round(pE[0],1))+","+str(np.round(pE[1],1))+"]"
            self.text = self.font.render(self.debug_text, True, (0, 0, 0), (255, 255, 255))
            self.window.blit(self.text, self.textRect)

        pygame.display.flip()    
        ##Slow down the loop to match FPS
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()