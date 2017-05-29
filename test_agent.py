# -*- coding: utf-8 -*-

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model",help="model file")
parser.add_argument("-m","--movie",default=None,help="monitor")
parser.add_argument("-r","--random",action="store_true",help="random action")
parser.add_argument("-i","--image",default=None,help="capture images")
parser.add_argument("-q","--qplot",action="store_true",help="plot Q values")
args = parser.parse_args()

from main_breakout import *
import cv2

ENV_NAME = "Breakout-v0"
FRAME_NUM = 4	# 状態を構成するフレーム数
FRAME_SKIP = 4

QPLOT = args.qplot
MOVIE = args.movie is not None
if MOVIE: MOV_DIR = args.movie

RANDOM = args.random

IMAGE = args.image is not None
if IMAGE: IMG_DIR = args.image

env = Environment(ENV_NAME,FRAME_SKIP,FRAME_NUM)
env.timestep = 0
env.test_mode = True

agent = Agent(env.action_space)
if not RANDOM: agent.load_model(args.model)
agent.test_mode = True

if QPLOT:
	out_file = open("plot_q_"+ENV_NAME+".csv","w")
	out_file.write(ENV_NAME+"\nTimestep,V(s)\n")

if MOVIE:
	env.to_movie_mode("./"+args.movie)
	
total_timestep = 0
for episode in xrange(1):
	print "Episode {:d}:".format(episode)
	
	#state = env.next_random_game()
	env.env.reset()
	
	for _ in xrange(env.frame_num):
		state = env.step(0)[0]
	

	done = False
	total_reward = 0.0
	timestep = 0
	while not done:
		timestep+=1
		total_timestep+=1
		
		if total_timestep % 100 == 0: print total_timestep
		
		if QPLOT:
			qvalue = agent.get_Q_values(state)
			max_q = np.max(qvalue)
			out_file.write("{:d},{:f}\n".format(timestep,max_q))

		action = random.choice(env.action_space) if RANDOM else agent.select_action(state)
		state,r,done,raw_obs = env.step(action,return_obs=True)
		total_reward+=r
		
		if IMAGE:
			cv2.imwrite(("{}/obs{:05d}.png").format(IMG_DIR,total_timestep),raw_obs[:,:,::-1])
	
	if QPLOT: out_file.close()

	print "  timestep={:d},total_reward={:.2f}".format(timestep,total_reward)
