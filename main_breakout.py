# -*- coding: utf-8 -*-
# Breakout

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import cv2
import gym
from gym import wrappers

import argparse
import random
import math
import copy
import time
import datetime
from collections import deque


class DeepQNetwork:
	def __init__(self):
		start_time = time.time()
	
		self.env = Environment(ENV_NAME,FRAME_SKIP,FRAME_NUM)
		
		self.agent = Agent(self.env.action_space)

		# ランダム行動でreplay memoryを貯める	
		print "Filling replay memory..."
		total_timestep = 1
		
		while total_timestep < LEARN_START:
			pre_state = self.env.next_random_game()
			
			timestep = 0

			done = False
			while not done:
				if total_timestep % 10000 == 0:
					print total_timestep
					
				action = random.choice(self.env.action_space)
				state,reward,done = self.env.step(action)
				self.store_experience(pre_state,action,reward,state,done)
				pre_state = state

				timestep+=1
				total_timestep+=1
		
		result_data = open("result_"+ENV_NAME+".csv","w")
		today = datetime.datetime.now()
		result_data.write("{}\n{}\nEpisode,Time[s],Total_timestep,Average_timestep,timestep_std,Average_reward,reward_std,Median_reward,Max_reward,Min_reward,,pdiff1,pdiff2,pdiff3\n".format(today.strftime("%Y/%m/%d %H:%M:%S"),ENV_NAME))

		print "Start Learning (total_time={:.2f}sec)".format(time.time() - start_time)

		# 学習開始
		total_timestep = 0
		episode = 0
		test_flag = True
		best_average_reward = -10000.0
		while total_timestep < MAX_TIMESTEP:
			episode+=1
			pre_state = self.env.next_random_game()
			
			timestep = 0
			done = False
			total_reward = 0.0
			while not done:
				action = self.agent.select_action(pre_state)
				state,reward,done = self.env.step(action)
				self.store_experience(pre_state,action,reward,state,done)
				pre_state = state
				
				total_reward+=reward
		
				self.agent.update()
				
				if total_timestep % TEST_INTERVAL == 0:
					test_flag = True

				timestep+=1
				total_timestep+=1
			
			total_time = time.time() - start_time
			
			print "Episode {:d}\n  total_reward={:.0f},timestep={:d},eps={:.5f},total_timestep={:d},total_time={:.2f}sec".format(episode,total_reward,timestep,self.agent.epsilon,total_timestep,total_time)

			if test_flag:
				self.agent.save_model("dqn_"+ENV_NAME+".model")
				result = DeepQNetwork.test_agent(self.env,self.agent)
				if result["average_reward"] > best_average_reward:
					best_average_reward = result["average_reward"]
					self.agent.save_model("dqn_"+ENV_NAME+"_best.model")
				
				result_data.flush()
				
				test_flag = False
				
		result_data.close()
		
	def store_experience(self,pre_state,action,reward,state,done):
		if len(pre_state) >= FRAME_NUM and len(state) >= FRAME_NUM:
			exp = pre_state, action, reward, state, done
			self.agent.memory.push_exp(exp)

	@classmethod
	def test_agent(cls,env,agent):
		TEST_EPISODE_NUM = 10
		print "Testing agent..."

		agent.test_mode = True
		env.test_mode = True
		
		reward_list = []
		timestep_list = []
		for episode in xrange(1,TEST_EPISODE_NUM+1):
			state = env.next_random_game()
			
			done = False
			timestep = 0
			total_reward = 0.0
			while not done:
				timestep+=1
				
				action = agent.select_action(state)
				state,reward,done = env.step(action)
				total_reward+=reward
			
			print "(Test) Episode {:d}: timestep={:d},total_reward={:.0f}".format(episode,timestep,total_reward)
			reward_list.append(total_reward)
			timestep_list.append(timestep)
			
			
		timestep_list = np.array(timestep_list,dtype=np.float64)
		reward_list = np.array(reward_list,dtype=np.float64)
		
		average_timestep = np.mean(timestep_list)
		average_reward = np.mean(reward_list)
		timestep_std = np.std(timestep_list)
		reward_std = np.std(reward_list)
		median_reward = np.median(reward_list)
		max_reward = np.max(reward_list)
		min_reward = np.min(reward_list)
		print "  average_timestep={:.0f}(std={:.0f}),average_reward={:.2f}(std={:.2f}),median_reward={:.1f}".format(average_timestep,timestep_std,average_reward,reward_std,median_reward)
		
		agent.test_mode = False
		env.test_mode = False
		
		ret = {}
		ret["average_timestep"] = average_timestep
		ret["average_reward"] = average_reward
		ret["timestep_std"] = timestep_std
		ret["reward_std"] = reward_std
		ret["median_reward"] = median_reward
		ret["max_reward"] = max_reward
		ret["min_reward"] = min_reward
		return ret
		

OBS_SIZE = 84
ZERO_OBS = np.zeros([OBS_SIZE,OBS_SIZE],dtype=np.uint8)
BLK_SCR = np.zeros([210,160,3],dtype=np.uint8)
EPISODE_MAX_LEN = 50000	# 1エピソードの最大長（超えたら打ち切り）
class Environment:
	def __init__(self,name,frameskip,frame_num,rendering=False):
		self.env = gym.make(name)
		self.env.frameskip = 1
		self.frameskip = frameskip
		self.frame_num = frame_num
		self.__buffer = deque(maxlen=frame_num)
		self.rendering = rendering
		self.done = False
		self.test_mode = False
		self.__obs_before = BLK_SCR
		self.movie_mode = False
		
		self.__action_space = [0,1,2,3]	# 0=NO-OP, 1=FIRE, 2=LEFT, 4=RIGHT
		self.action_space = range(len(self.__action_space))

	def new_game(self,return_obs=False):
		self.timestep = 0
	
		if self.done:
			self.done = False
		else:
			lives = self.env.ale.lives()
			while lives >= 1:
				# 中途半端な時は今のエピソードが終わるまで回す
				self.env.step(random.choice(self.__action_space))
				if self.env.ale.lives() < lives: break
			
		# ライフが0またはテスト時はライフMAXの状態に戻す
		if self.test_mode or self.env.ale.lives() == 0:
			self.env.reset()
		
		self.__obs_before = BLK_SCR
		raw_obs = self.__step(0)[0]
		
		return (list(self.__buffer),raw_obs) if return_obs else list(self.__buffer)
	
	def next_random_game(self,return_obs=False):
		raw_obs = self.new_game(return_obs=True)[1]
		
		# 初期状態をランダムにする
		for _ in xrange(random.randint(0,30)):
			raw_obs = self.__step(0)[0]
		
		return (list(self.__buffer),raw_obs) if return_obs else list(self.__buffer)
		
	def step(self,action_index,return_obs=False):
		if self.movie_mode:
			ret = self.__test_step(action_index,return_obs)
			return ret
	
		self.timestep+=1
		action = self.__action_space[action_index]
		
		lives = self.env.ale.lives()
		reward = 0.0
		for _ in xrange(self.frameskip):
			raw_obs,r,d = self.__step(action)
			reward+=r
			if d: break
		observation = Environment.__preprocess(raw_obs)
		
		if ((not self.test_mode) and self.env.ale.lives() < lives) or (self.test_mode and self.env.ale.lives() == 0) or self.timestep >= EPISODE_MAX_LEN:
			# 訓練時は1ミスでエピソード終了
			# テスト時はライフが無くなったらエピソード終了
			self.done = True
		
		self.__buffer.append(observation)
		
		if return_obs:
			return list(self.__buffer),reward,self.done,raw_obs
		else:
			return list(self.__buffer),reward,self.done
			
	# movie録画用のstep()
	def __test_step(self,action_index,return_obs=False):
		self.timestep+=1
		action = self.__action_space[action_index]
		
		reward = 0.0
		for _ in xrange(self.frameskip):
			raw_obs,r,done = self.__step(action)
			reward+=r
			if done: break
		observation = Environment.__preprocess(raw_obs)
		
		self.__buffer.append(observation)
		
		if return_obs:
			return list(self.__buffer),reward,done,raw_obs
		else:
			return list(self.__buffer),reward,done
			
	def __step(self,action):
		obs,reward,done,i = self.env.step(action)
		ret_obs = np.maximum(obs,self.__obs_before)
		self.__obs_before = obs
		
		if self.rendering: self.env.render()
		return ret_obs,reward,done
	
	@classmethod
	def __preprocess(cls,_observation):
		# 前処理
		obs = cv2.resize(_observation,(OBS_SIZE,OBS_SIZE))
		obs = cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY)	#グレースケール化
		return obs
		
	def to_movie_mode(self,movie_dir):
		self.env = wrappers.Monitor(self.env,movie_dir,force=True)
		self.movie_mode = True
		
		return self
		

BATCH_SIZE = 32
Q_SYNC_INTERVAL = 10000
UPDATE_FREQ = 4

class Agent:
	TEST_EPSILON = 0.05

	def __init__(self,actions):
		self.test_mode = False
		
		self.__Q = QModel(len(actions))
		if GPU_MODE: self.__Q.to_gpu()
		self.__Q_target = copy.deepcopy(self.__Q)
		self.__Q_init = copy.deepcopy(self.__Q)
		
		self.memory = ReplayMemory()
		
		self.__actions = actions
		
		self.__epsilon = 1.0
		self.__gamma = 0.99
		
		self.__optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.0, eps=0.01)
		self.__optimizer.use_cleargrads()
		self.__optimizer.setup(self.__Q)
		
		self.__timestep = 0

	def select_action(self,state):
		eps = Agent.TEST_EPSILON if self.test_mode else self.__epsilon
		
		if np.random.rand() < eps:
			# random
			a = random.choice(self.__actions)
		elif len(state) >= FRAME_NUM:
			# greedy
			qvalue = self.get_Q_values(state)
			besta_index = np.where(qvalue==qvalue.max())[0]	#最大値が複数あった場合はその中からランダムに選ぶ
			a = np.random.choice(besta_index)
		else:
			# 4フレーム分ない時は0を返す
			a = 0
			
		return a
		
	def get_Q_values(self,state):
		_state = xp.array(state,dtype=np.float32) / 255.0
		_state = xp.expand_dims(_state,axis=0)
		qvalue = self.__Q(Variable(_state,volatile="on")).data
		qvalue = xp.reshape(qvalue,[len(self.__actions)])
		if GPU_MODE:
			qvalue = xp.asnumpy(qvalue)
			
		return qvalue
		
	def update(self):
		self.__timestep+=1
		if self.__timestep % UPDATE_FREQ == 0:
			self.learn()
		if self.__timestep % Q_SYNC_INTERVAL == 0:
			self.__Q_target = copy.deepcopy(self.__Q)
			print "Target updated."
			
		self.update_epsilon()
		
	def learn(self):
		if self.test_mode: return
		
		batch = self.memory.get_batch(BATCH_SIZE)
		pre_state,action,reward,state,done = [],[],[],[],[]
		for b in batch:
			pre_state.append(b[0])
			action.append(b[1])
			reward.append(b[2])
			state.append(b[3])
			done.append(b[4])
		
		done = xp.array(done,dtype=np.float32)	# True=1.0, False=0.0
		action = np.array(action,dtype=np.uint8)
		pre_state = Variable(xp.array(pre_state,dtype=np.float32) / 255.0)
		state = Variable(xp.array(state,dtype=np.float32) / 255.0)
		reward = xp.sign(xp.array(reward,dtype=np.float32))
		
		q_value = self.__Q(pre_state)		# backwardあり
		
		q_target = self.__Q_target(state).data	# 定数
		q_target_max = xp.max(q_target,axis=1)
		
		# targetを作る
		actions_one_hot = np.zeros([BATCH_SIZE,len(self.__actions)],dtype=np.float32)
		actions_one_hot[np.arange(BATCH_SIZE),action] = 1.0	#cupyだとできない
		if GPU_MODE:
			actions_one_hot = cuda.to_gpu(actions_one_hot)
		t1 = reward + (1.0 - done) * self.__gamma * q_target_max
		t1 = t1.reshape([BATCH_SIZE,1])
		t1 = actions_one_hot * t1
		t2 = (1.0 - actions_one_hot) * q_value.data	# 取った行動以外の部分はQ値で埋める
		target = t1 + t2
		
		self.__Q.cleargrads()
		
		# 損失計算
		loss = clipped_loss(Variable(target),q_value)
		loss.backward()
		self.__optimizer.update()
		
	def update_epsilon(self):
		# 100万イテレーションで0.1まで減少させる
		self.__epsilon-=0.9e-6
		self.__epsilon = max(0.1,self.__epsilon)

	@property
	def epsilon(self):
		return self.__epsilon
		
	def save_model(self,filename):
		serializers.save_npz(filename,self.__Q)
		print "Model saved: "+filename
		
	def load_model(self,filename):
		serializers.load_npz(filename,self.__Q)
		print "Model loaded: "+filename
	
def clipped_loss(target,q_value):
	error_abs = F.absolute(target - q_value)
	quadratic_part = F.clip(error_abs,0.0,1.0)
	linear_part = error_abs - quadratic_part
	loss = F.sum(0.5 * F.square(quadratic_part) + linear_part)
	return loss

class QModel(Chain):
	def __init__(self,output_num):
		INPUT_CH = 4
		CONV1_OUT = 32
		CONV2_OUT = 64
		CONV3_OUT = 64
		FC1_IN = 3136
		FC1_OUT = 512

		super(QModel,self).__init__(
			conv1 = convlayer(INPUT_CH,CONV1_OUT,k=8,s=4,p=1),
			conv2 = convlayer(CONV1_OUT,CONV2_OUT,k=4,s=2),
			conv3 = convlayer(CONV2_OUT,CONV3_OUT,k=3,s=1),
			fc1 = linear(FC1_IN,FC1_OUT),
			fc2 = linear(FC1_OUT,output_num)
		)

	def __call__(self,x):
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		h = F.relu(self.fc1(h))
		h = self.fc2(h)
		return h
		
def convlayer(input,output,k,s,p=0):
	std = 1.0/(np.sqrt(input)*k)
	w = np.random.uniform(-std,std,[output,input,k,k]).astype(np.float32)
	b = np.random.uniform(-std,std,[output]).astype(np.float32)
	return L.Convolution2D(input,output,ksize=k,stride=s,pad=p,initialW=w,initial_bias=b)
	
def linear(input,output):
	std = 1.0/np.sqrt(input)
	w = np.random.uniform(-std,std,[output,input]).astype(np.float32)
	b = np.random.uniform(-std,std,[output]).astype(np.float32)
	return L.Linear(input,output,initialW=w,initial_bias=b)

class ReplayMemory:
	def __init__(self):
		self.__memory = deque(maxlen=MEM_SIZE)
		
	def push_exp(self,exp):	
		self.__memory.append(exp)
		
	def get_batch(self,batch_size):
		return random.sample(self.__memory,batch_size)
		
		
ENV_NAME = "Breakout-v0"
FRAME_NUM = 4	# 状態を構成するフレーム数
FRAME_SKIP = 4
MAX_TIMESTEP = 50000000	# イテレーション回数
TEST_INTERVAL = 250000
LEARN_START = 50000
MEM_SIZE = 1000000

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=ENV_NAME)
	parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU ID")
	parser.add_argument("--learn_start",type=int,default=50000)
	parser.add_argument("--test_interval",type=int,default=250000)
	parser.add_argument("-i","--iteration",type=int,default=50000000,"number of iterations")
	parser.add_argument("--mem",type=int,default=1000000,help="capacity of replay memory")
		
	# GPUが使えるか確認
	args = parser.parse_args()
	GPU_MODE = args.gpu >= 0
	if GPU_MODE:
		cuda.check_cuda_available()
		cuda.get_device(args.gpu).use()
		print "GPU mode"
		xp = cuda.cupy
	else:
		print "CPU mode"
		xp = np
	
	FRAME_NUM = 4	# 状態を構成するフレーム数
	FRAME_SKIP = 4
	MAX_TIMESTEP = args.iteration	# イテレーション回数
	TEST_INTERVAL = args.test_interval
	LEARN_START = args.learn_start
	MEM_SIZE = args.mem

	DeepQNetwork()
else:
	GPU_MODE = False
	xp = np

