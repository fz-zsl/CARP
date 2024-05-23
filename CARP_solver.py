import argparse
import math
import multiprocessing
import random
import time
from copy import deepcopy

import numpy as np


class Config(object):
	def __init__(self, p_fl=.25, p_si=.05, p_di=.05, p_sw=.15, p_2o=0.35, p_ms=None):
		self.p_fl = p_fl
		self.p_si = p_si
		self.p_di = p_di
		self.p_sw = p_sw
		self.p_2o = p_2o
		self.p_ms = (1.0 - p_fl - p_si - p_di - p_sw - p_2o) if p_ms is None else p_ms

		assert self.p_2o >= .0
		assert abs(self.p_fl + self.p_si + self.p_di + self.p_sw + self.p_2o + self.p_ms - 1.0) < 0.0001

		self.dist = [
			p_fl,
			p_fl + p_si,
			p_fl + p_si + p_di,
			p_fl + p_si + p_di + p_sw,
			p_fl + p_si + p_di + p_sw + p_2o,
			1.0
		]


class Edge:
	def __init__(self, fr, to, cost, demand):
		self.fr = fr
		self.to = to
		self.cost = cost
		self.demand = demand

	def __str__(self):
		return f"({self.fr}, {self.to}, {self.cost}, {self.demand})"

	def __repr__(self):
		return f"({self.fr}, {self.to}, {self.cost}, {self.demand})"


def load_from_file(filename):
	with open(filename, 'r') as f:
		data = f.readlines()
	info = {
		'name': None,
		'vertices': None,
		'edges_req': None,
		'edges_no_req': None,
		'capacity': None,
		'depot': None,
		'edges': []
	}
	for line in data:
		line = line.replace(',', ' ,').strip().split()
		if not line:
			continue
		if line[0] == 'NAME':
			info['name'] = line[2]
		elif line[0] == 'VERTICES':
			info['vertices'] = int(line[2])
		elif line[0] == 'REQUIRED':
			info['edges_req'] = int(line[3])
		elif line[0] == 'NON-REQUIRED':
			info['edges_no_req'] = int(line[3])
		elif line[0] == 'CAPACITY':
			info['capacity'] = int(line[2])
		elif line[0] == 'DEPOT':
			info['depot'] = int(line[2])
		elif line[0].isdigit():
			info['edges'].append(Edge(
				int(line[0]), int(line[1]), int(line[2]), int(line[3])
			))
	return info


EDGES = []
DIS = np.full((1, 1), np.inf)


def rint(a, b):
	if a == b:
		return a
	return random.randint(a, b)


class CMO:
	@staticmethod
	def flip(problem, T):
		route_id = rint(0, problem.cnt - 1)
		while len(problem.routes[route_id]) < 2:
			route_id = rint(0, problem.cnt - 1)
		edge_id = rint(0, len(problem.routes[route_id]) - 1)
		delta_cost = -DIS[
			problem.depot if edge_id == 0 else problem.routes[route_id][edge_id - 1].to,
			problem.routes[route_id][edge_id].fr
		]
		delta_cost -= DIS[
			problem.routes[route_id][edge_id].to,
			problem.depot if edge_id == len(problem.routes[route_id]) - 1 else problem.routes[route_id][edge_id + 1].fr
		]
		edge = problem.routes[route_id][edge_id]
		problem.routes[route_id][edge_id] = Edge(edge.to, edge.fr, edge.cost, edge.demand)
		delta_cost += DIS[
			problem.depot if edge_id == 0 else problem.routes[route_id][edge_id - 1].to,
			problem.routes[route_id][edge_id].fr
		]
		delta_cost += DIS[
			problem.routes[route_id][edge_id].to,
			problem.depot if edge_id == len(problem.routes[route_id]) - 1 else problem.routes[route_id][edge_id + 1].fr
		]
		if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):  # accept the move
			problem.tot_cost += delta_cost
			return
		# undo the move
		edge = problem.routes[route_id][edge_id]
		problem.routes[route_id][edge_id] = Edge(edge.to, edge.fr, edge.cost, edge.demand)

	@staticmethod
	def single_insertion(problem, T):
		route_id = rint(0, problem.cnt - 1)
		same_route = random.random() < 0.45
		_cnt = 0
		while len(problem.routes[route_id]) < 2:
			route_id = rint(0, problem.cnt - 1)
			_cnt += 1
			if _cnt > 10:
				return
		edge_id = rint(0, len(problem.routes[route_id]) - 1)
		edge = deepcopy(problem.routes[route_id][edge_id])
		ins_id = route_id if same_route else rint(0, problem.cnt - 1)
		ins_edge_id = rint(0, len(problem.routes[ins_id]) - (1 if route_id == ins_id else 0))
		if ins_id != route_id and problem.caps[ins_id] + edge.demand > problem.capacity:
			return

		delta_cost = -DIS[
			problem.depot if edge_id == 0 else problem.routes[route_id][edge_id - 1].to,
			edge.fr
		]
		delta_cost -= DIS[
			edge.to,
			problem.depot if edge_id == len(problem.routes[route_id]) - 1 else problem.routes[route_id][edge_id + 1].fr
		]
		delta_cost += DIS[
			problem.depot if edge_id == 0 else problem.routes[route_id][edge_id - 1].to,
			problem.depot if edge_id == len(problem.routes[route_id]) - 1 else problem.routes[route_id][edge_id + 1].fr
		]
		problem.caps[route_id] -= edge.demand
		problem.caps[ins_id] += edge.demand
		problem.routes[route_id].pop(edge_id)
		problem.routes[ins_id].insert(ins_edge_id, edge)
		delta_cost += DIS[
			problem.depot if ins_edge_id == 0 else problem.routes[ins_id][ins_edge_id - 1].to,
			edge.fr
		]
		delta_cost += DIS[
			edge.to,
			problem.depot if ins_edge_id == len(problem.routes[ins_id]) - 1 else problem.routes[ins_id][ins_edge_id + 1].fr
		]
		delta_cost -= DIS[
			problem.depot if ins_edge_id == 0 else problem.routes[ins_id][ins_edge_id - 1].to,
			problem.depot if ins_edge_id == len(problem.routes[ins_id]) - 1 else problem.routes[ins_id][ins_edge_id + 1].fr
		]
		if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):  # accept the move
			problem.tot_cost += delta_cost
			return
		# undo the move
		problem.caps[route_id] += edge.demand
		problem.routes[ins_id].pop(ins_edge_id)
		problem.routes[route_id].insert(edge_id, edge)
		problem.caps[ins_id] -= edge.demand

	@staticmethod
	def double_insertion(problem, T):
		route_id = rint(0, problem.cnt - 1)
		same_route = random.random() < 0.45
		_cnt = 0
		while len(problem.routes[route_id]) < 3:
			route_id = rint(0, problem.cnt - 1)
			_cnt += 1
			if _cnt > 10:
				return
		edge_id = rint(0, len(problem.routes[route_id]) - 2)
		ins_id = route_id if same_route else rint(0, problem.cnt - 1)
		ins_edge_id = rint(0, len(problem.routes[ins_id]) - (2 if route_id == ins_id else 0))
		edge1 = deepcopy(problem.routes[route_id][edge_id])
		edge2 = deepcopy(problem.routes[route_id][edge_id + 1])

		delta_cost = -DIS[
			problem.depot if edge_id == 0 else problem.routes[route_id][edge_id - 1].to,
			edge1.fr
		]
		delta_cost -= DIS[
			edge2.to,
			problem.depot if edge_id == len(problem.routes[route_id]) - 2 else problem.routes[route_id][edge_id + 2].fr
		]
		delta_cost += DIS[
			problem.depot if edge_id == 0 else problem.routes[route_id][edge_id - 1].to,
			problem.depot if edge_id == len(problem.routes[route_id]) - 2 else problem.routes[route_id][edge_id + 2].fr
		]
		if ins_id != route_id and problem.caps[ins_id] + edge1.demand + edge2.demand > problem.capacity:
			return
		problem.caps[route_id] -= edge1.demand + edge2.demand
		problem.caps[ins_id] += edge1.demand + edge2.demand
		problem.routes[route_id].pop(edge_id)
		problem.routes[route_id].pop(edge_id)
		problem.routes[ins_id].insert(ins_edge_id, edge1)
		problem.routes[ins_id].insert(ins_edge_id + 1, edge2)

		delta_cost += DIS[
			problem.depot if ins_edge_id == 0 else problem.routes[ins_id][ins_edge_id - 1].to,
			edge1.fr
		]
		delta_cost += DIS[
			edge2.to,
			problem.depot if ins_edge_id == len(problem.routes[ins_id]) - 2 else problem.routes[ins_id][ins_edge_id + 2].fr
		]
		delta_cost -= DIS[
			problem.depot if ins_edge_id == 0 else problem.routes[ins_id][ins_edge_id - 1].to,
			problem.depot if ins_edge_id == len(problem.routes[ins_id]) - 2 else problem.routes[ins_id][ins_edge_id + 2].fr
		]
		if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):  # accept the move
			problem.tot_cost += delta_cost
			return
		# undo the move
		problem.caps[route_id] += edge1.demand + edge2.demand
		problem.caps[ins_id] -= edge1.demand + edge2.demand
		problem.routes[ins_id].pop(ins_edge_id)
		problem.routes[ins_id].pop(ins_edge_id)
		problem.routes[route_id].insert(edge_id, edge1)
		problem.routes[route_id].insert(edge_id + 1, edge2)

	@staticmethod
	def swap(problem, T):
		route_id1 = 0
		route_id2 = 0
		edge_id1 = 0
		edge_id2 = 0
		same_route = (random.random() < 0.45)
		while route_id1 == route_id2 and edge_id1 == edge_id2 \
				or len(problem.routes[route_id1]) == 0 \
				or len(problem.routes[route_id2]) == 0:
			route_id1 = rint(0, problem.cnt - 1)
			route_id2 = route_id1 if same_route else rint(0, problem.cnt - 1)
			edge_id1 = rint(0, len(problem.routes[route_id1]) - 1) if len(
				problem.routes[route_id1]) > 1 else 0
			edge_id2 = rint(0, len(problem.routes[route_id2]) - 1) if len(
				problem.routes[route_id2]) > 1 else 0
		if edge_id1 > edge_id2:
			edge_id1, edge_id2 = edge_id2, edge_id1
			route_id1, route_id2 = route_id2, route_id1
		edge1 = deepcopy(problem.routes[route_id1][edge_id1])
		edge2 = deepcopy(problem.routes[route_id2][edge_id2])

		if route_id1 == route_id2 and edge_id2 - edge_id1 == 1:
			delta_cost = -DIS[
				problem.depot if edge_id1 == 0 else problem.routes[route_id1][edge_id1 - 1].to,
				edge1.fr
			]
			delta_cost -= DIS[
				edge1.to,
				edge2.fr
			]
			delta_cost -= DIS[
				edge2.to,
				problem.depot if edge_id2 == len(problem.routes[route_id1]) - 1 else problem.routes[route_id1][edge_id2 + 1].fr
			]
		else:
			delta_cost = -DIS[
				problem.depot if edge_id1 == 0 else problem.routes[route_id1][edge_id1 - 1].to,
				edge1.fr
			]
			delta_cost -= DIS[
				edge1.to,
				problem.depot if edge_id1 == len(problem.routes[route_id1]) - 1 else problem.routes[route_id1][edge_id1 + 1].fr
			]
			delta_cost -= DIS[
				problem.depot if edge_id2 == 0 else problem.routes[route_id2][edge_id2 - 1].to,
				edge2.fr
			]
			delta_cost -= DIS[
				edge2.to,
				problem.depot if edge_id2 == len(problem.routes[route_id2]) - 1 else problem.routes[route_id2][
					edge_id2 + 1].fr
			]

		if route_id1 != route_id2 and \
				(problem.caps[route_id1] + edge2.demand - edge1.demand > problem.capacity
				or problem.caps[route_id2] + edge1.demand - edge2.demand > problem.capacity):
			return
		problem.caps[route_id1] += edge2.demand - edge1.demand
		problem.caps[route_id2] += edge1.demand - edge2.demand
		problem.routes[route_id1][edge_id1] = edge2
		problem.routes[route_id2][edge_id2] = edge1

		if route_id1 == route_id2 and edge_id2 - edge_id1 == 1:
			delta_cost += DIS[
				problem.depot if edge_id1 == 0 else problem.routes[route_id1][edge_id1 - 1].to,
				edge2.fr
			]
			delta_cost += DIS[
				edge2.to,
				edge1.fr
			]
			delta_cost += DIS[
				edge1.to,
				problem.depot if edge_id2 == len(problem.routes[route_id1]) - 1 else problem.routes[route_id1][edge_id2 + 1].fr
			]
		else:
			delta_cost += DIS[
				problem.depot if edge_id1 == 0 else problem.routes[route_id1][edge_id1 - 1].to,
				edge2.fr
			]
			delta_cost += DIS[
				edge2.to,
				problem.depot if edge_id1 == len(problem.routes[route_id1]) - 1 else problem.routes[route_id1][edge_id1 + 1].fr
			]
			delta_cost += DIS[
				problem.depot if edge_id2 == 0 else problem.routes[route_id2][edge_id2 - 1].to,
				edge1.fr
			]
			delta_cost += DIS[
				edge1.to,
				problem.depot if edge_id2 == len(problem.routes[route_id2]) - 1 else problem.routes[route_id2][edge_id2 + 1].fr
			]
		if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):  # accept the move
			problem.tot_cost += delta_cost
			return
		# undo the move
		problem.caps[route_id1] -= edge2.demand - edge1.demand
		problem.caps[route_id2] -= edge1.demand - edge2.demand
		problem.routes[route_id1][edge_id1] = edge1
		problem.routes[route_id2][edge_id2] = edge2

	@staticmethod
	def opt(problem, T):
		route_id1 = -1
		route_id2 = -1
		same_route = (random.random() < 0.45)
		_cnt = 0
		while route_id1 < 0 or len(problem.routes[route_id1]) < 2 or len(problem.routes[route_id2]) < 2:
			route_id1 = rint(0, problem.cnt - 1)
			route_id2 = route_id1 if same_route else rint(0, problem.cnt - 1)
			_cnt += 1
			if _cnt > 10:
				return

		if route_id1 != route_id2:
			edge_id1 = rint(0, len(problem.routes[route_id1])) if len(problem.routes[route_id1]) > 0 else 0
			edge_id2 = rint(0, len(problem.routes[route_id2])) if len(problem.routes[route_id2]) > 0 else 0
			prev_route1 = problem.routes[route_id1]
			prev_route2 = problem.routes[route_id2]

			delta_cost = -DIS[
				prev_route1[edge_id1 - 1].to if edge_id1 > 0 else problem.depot,
				prev_route1[edge_id1].fr if edge_id1 < len(prev_route1) else problem.depot
			]
			delta_cost -= DIS[
				prev_route2[edge_id2 - 1].to if edge_id2 > 0 else problem.depot,
				prev_route2[edge_id2].fr if edge_id2 < len(prev_route2) else problem.depot
			]
			caps1a = sum(_.demand for _ in prev_route1[:edge_id1])
			caps1b = sum(_.demand for _ in prev_route1[edge_id1:])
			caps2a = sum(_.demand for _ in prev_route2[:edge_id2])
			caps2b = sum(_.demand for _ in prev_route2[edge_id2:])
			flag1 = (caps1a + caps2b <= problem.capacity and caps2a + caps1b <= problem.capacity)
			flag2 = (caps1a + caps2a <= problem.capacity and caps1b + caps2b <= problem.capacity)
			if (not flag1) and (not flag2):
				return
			_cost1 = DIS[
				prev_route1[edge_id1 - 1].to if edge_id1 > 0 else problem.depot,
				prev_route2[edge_id2].fr if edge_id2 < len(prev_route2) else problem.depot
			]
			_cost1 += DIS[
				prev_route2[edge_id2 - 1].to if edge_id2 > 0 else problem.depot,
				prev_route1[edge_id1].fr if edge_id1 < len(prev_route1) else problem.depot
			]
			_cost2 = DIS[
				prev_route1[edge_id1 - 1].to if edge_id1 > 0 else problem.depot,
				prev_route2[edge_id2 - 1].to if edge_id2 > 0 else problem.depot
			]
			_cost2 += DIS[
				prev_route1[edge_id1].fr if edge_id1 < len(prev_route1) else problem.depot,
				prev_route2[edge_id2].fr if edge_id2 < len(prev_route2) else problem.depot
			]
			if flag1 and (not flag2):
				choice = 1
			elif (not flag1) and flag2:
				choice = 2
			else:
				choice = 1 if _cost1 <= _cost2 else 2
			delta_cost += _cost1 if choice == 1 else _cost2
			if delta_cost >= 0 and random.random() >= math.exp(-delta_cost / T):
				return
			problem.tot_cost += delta_cost
			if choice == 1:
				problem.routes[route_id1], problem.routes[route_id2] = \
					(problem.routes[route_id1][:edge_id1] + problem.routes[route_id2][edge_id2:],
					 problem.routes[route_id2][:edge_id2] + problem.routes[route_id1][edge_id1:])
				problem.caps[route_id1] = caps1a + caps2b
				problem.caps[route_id2] = caps2a + caps1b
			else:
				new_route1 = problem.routes[route_id1][:edge_id1] + \
				             [Edge(_.to, _.fr, _.cost, _.demand) for _ in problem.routes[route_id2][:edge_id2][::-1]]
				new_route2 = [Edge(_.to, _.fr, _.cost, _.demand) for _ in problem.routes[route_id2][edge_id2:][::-1]] + \
				             problem.routes[route_id1][edge_id1:]
				problem.routes[route_id1], problem.routes[route_id2] = new_route1, new_route2
				problem.caps[route_id1] = caps1a + caps2a
				problem.caps[route_id2] = caps2b + caps1b
		else:
			while len(problem.routes[route_id1]) < 2:
				route_id1 = rint(0, problem.cnt - 1)
			edge_id1 = rint(0, len(problem.routes[route_id1]) - 1)
			edge_id2 = rint(0, len(problem.routes[route_id1]) - 1)
			if edge_id1 > edge_id2:
				edge_id1, edge_id2 = edge_id2, edge_id1
			edge_id2 += 1  # [edge1, edge2)
			delta_cost = -DIS[
				problem.routes[route_id1][edge_id1 - 1].to if edge_id1 > 0 else problem.depot,
				problem.routes[route_id1][edge_id1].fr
			]
			delta_cost -= DIS[
				problem.routes[route_id1][edge_id2 - 1].to,
				problem.routes[route_id1][edge_id2].fr if edge_id2 < len(problem.routes[route_id1]) else problem.depot
			]
			delta_cost += DIS[
				problem.routes[route_id1][edge_id1 - 1].to if edge_id1 > 0 else problem.depot,
				problem.routes[route_id1][edge_id2 - 1].to
			]
			delta_cost += DIS[
				problem.routes[route_id1][edge_id1].fr,
				problem.routes[route_id1][edge_id2].fr if edge_id2 < len(problem.routes[route_id1]) else problem.depot
			]
			if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
				sub_route = problem.routes[route_id1][edge_id1:edge_id2]
				problem.routes[route_id1][edge_id1:edge_id2] = \
					[Edge(_.to, _.fr, _.cost, _.demand) for _ in sub_route[::-1]]
				problem.tot_cost += delta_cost

	@staticmethod
	def merge_split(problem, T):
		MS_PROB = 0.2
		ms_routes = random.sample(range(0, problem.cnt), math.ceil(problem.cnt * MS_PROB))
		new_routes = []
		new_caps = []
		free = []
		for route_id in range(problem.cnt):
			if route_id in ms_routes:
				for edge in problem.routes[route_id]:
					free.append(edge)
			else:
				# print(problem.cnt,len(problem.routes), route_id)
				new_routes.append(problem.routes[route_id])
				new_caps.append(problem.caps[route_id])
		random.shuffle(free)
		new_cnt = len(new_routes)
		new_routes.append([])
		new_caps.append(0)
		remain = problem.capacity
		endpoint = problem.depot
		while free:
			min_dis = np.inf
			min_e = None
			mode = problem.mode
			for e in free:
				if e.demand > remain:
					continue
				_dis = min(float(DIS[endpoint, e.fr]), float(DIS[endpoint, e.to]))
				if _dis > min_dis:
					continue
				if _dis < min_dis:
					min_dis = _dis
					min_e = e
					continue
				res = False
				if DIS[endpoint, e.fr] < DIS[endpoint, e.to]:  # endpoint -> e.fr -> e.to
					res |= mode == 0 and DIS[e.to, problem.depot] > DIS[min_e.to, problem.depot]
					res |= mode == 1 and DIS[e.to, problem.depot] < DIS[min_e.to, problem.depot]
					res |= mode == 2 and e.demand / e.cost > min_e.demand / min_e.cost
					res |= mode == 3 and e.demand / e.cost < min_e.demand / min_e.cost
					res |= mode == 4 and remain > problem.capacity / 2 \
					       and DIS[e.to, problem.depot] > DIS[min_e.to, problem.depot]
					res |= mode == 4 and remain <= problem.capacity / 2 \
					       and DIS[e.to, problem.depot] < DIS[min_e.to, problem.depot]
				else:  # endpoint -> e.to -> e.fr
					res |= mode == 0 and DIS[e.fr, problem.depot] > DIS[min_e.fr, problem.depot]
					res |= mode == 1 and DIS[e.fr, problem.depot] < DIS[min_e.fr, problem.depot]
					res |= mode == 2 and e.demand / e.cost > min_e.demand / min_e.cost
					res |= mode == 3 and e.demand / e.cost < min_e.demand / min_e.cost
					res |= mode == 4 and remain > problem.capacity / 2 \
					       and DIS[e.fr, problem.depot] > DIS[min_e.fr, problem.depot]
					res |= mode == 4 and remain <= problem.capacity / 2 \
					       and DIS[e.fr, problem.depot] < DIS[min_e.fr, problem.depot]
				if res:
					min_e = e  # min_dis is the same
			if min_e is None:
				new_cnt += 1
				remain = problem.capacity
				endpoint = problem.depot
				new_routes.append([])
				new_caps.append(0)
			elif DIS[endpoint, min_e.fr] < DIS[endpoint, min_e.to]:
				new_routes[new_cnt].append(Edge(min_e.fr, min_e.to, min_e.cost, min_e.demand))
				new_caps[new_cnt] += min_e.demand
				remain -= min_e.demand
				endpoint = min_e.to
				free.remove(min_e)
			else:
				new_routes[new_cnt].append(Edge(min_e.to, min_e.fr, min_e.cost, min_e.demand))
				new_caps[new_cnt] += min_e.demand
				remain -= min_e.demand
				endpoint = min_e.fr
				free.remove(min_e)
		new_cost = problem.eval(new_routes)
		delta_cost = new_cost - problem.tot_cost
		if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
			problem.routes = new_routes
			problem.caps = new_caps
			problem.tot_cost = new_cost
			problem.cnt = new_cnt + 1


	@staticmethod
	def perform_cmo(problem, cfg, T):
		_ = random.random()
		for i in range(len(cfg.dist)):
			if _ >= cfg.dist[i]:
				continue
			if i == 0:
				CMO.flip(problem, T)
			elif i == 1:
				CMO.single_insertion(problem, T)
			elif i == 2:
				CMO.double_insertion(problem, T)
			elif i == 3:
				CMO.swap(problem, T)
			elif i == 4:
				CMO.opt(problem, T)
			else:
				CMO.merge_split(problem, T)
			# print(i)
			assert len(problem.routes) == problem.cnt
			break


class Problem:
	def __init__(self, filename, mode):
		_info = load_from_file(filename)
		self.name = str(_info['name'])
		self.n = int(_info['vertices'])
		self.m_req = int(_info['edges_req'])
		self.m_no_req = int(_info['edges_no_req'])
		self.m = self.m_req + self.m_no_req
		self.capacity = int(_info['capacity'])
		self.depot = int(_info['depot']) - 1
		self.cnt = 0
		self.routes = [[]]
		self.caps = [0]
		self.best_routes = None
		self.tot_cost = 0
		self.mode = mode
		self.best_cost = np.infty

		if mode == 0:  # Floyd-Warshall
			global EDGES
			global DIS
			EDGES = _info['edges']
			DIS = np.full((self.n, self.n), np.inf)
			for i in range(self.n):
				DIS[i, i] = 0
			for edge in EDGES:
				edge.fr -= 1
				edge.to -= 1
				DIS[edge.fr, edge.to] = edge.cost
				DIS[edge.to, edge.fr] = edge.cost
			for k in range(self.n):
				for i in range(self.n):
					for j in range(self.n):
						DIS[i, j] = min(float(DIS[i, j]), float(DIS[i, k] + DIS[k, j]))

	def path_scanning(self):
		free = [e for e in EDGES if e.demand > 0]
		if not free:
			return
		remain = self.capacity
		endpoint = self.depot
		while free:
			min_dis = np.inf
			min_e = None
			mode = self.mode
			for e in free:
				if e.demand > remain:
					continue
				_dis = min(float(DIS[endpoint, e.fr]), float(DIS[endpoint, e.to]))
				if _dis > min_dis:
					continue
				if _dis < min_dis:
					min_dis = _dis
					min_e = e
					continue
				res = False
				if DIS[endpoint, e.fr] < DIS[endpoint, e.to]:  # endpoint -> e.fr -> e.to
					res |= mode == 0 and DIS[e.to, self.depot] > DIS[min_e.to, self.depot]
					res |= mode == 1 and DIS[e.to, self.depot] < DIS[min_e.to, self.depot]
					res |= mode == 2 and e.demand / e.cost > min_e.demand / min_e.cost
					res |= mode == 3 and e.demand / e.cost < min_e.demand / min_e.cost
					res |= mode == 4 and remain > self.capacity / 2 \
					       and DIS[e.to, self.depot] > DIS[min_e.to, self.depot]
					res |= mode == 4 and remain <= self.capacity / 2 \
					       and DIS[e.to, self.depot] < DIS[min_e.to, self.depot]
				else:  # endpoint -> e.to -> e.fr
					res |= mode == 0 and DIS[e.fr, self.depot] > DIS[min_e.fr, self.depot]
					res |= mode == 1 and DIS[e.fr, self.depot] < DIS[min_e.fr, self.depot]
					res |= mode == 2 and e.demand / e.cost > min_e.demand / min_e.cost
					res |= mode == 3 and e.demand / e.cost < min_e.demand / min_e.cost
					res |= mode == 4 and remain > self.capacity / 2 \
					       and DIS[e.fr, self.depot] > DIS[min_e.fr, self.depot]
					res |= mode == 4 and remain <= self.capacity / 2 \
					       and DIS[e.fr, self.depot] < DIS[min_e.fr, self.depot]
				if res:
					min_e = e  # min_dis is the same
			if min_e is None:
				self.cnt += 1
				remain = self.capacity
				endpoint = self.depot
				self.routes.append([])
				self.caps.append(0)
			elif DIS[endpoint, min_e.fr] < DIS[endpoint, min_e.to]:
				self.routes[self.cnt].append(Edge(min_e.fr, min_e.to, min_e.cost, min_e.demand))
				self.caps[self.cnt] += min_e.demand
				remain -= min_e.demand
				endpoint = min_e.to
				free.remove(min_e)
			else:
				self.routes[self.cnt].append(Edge(min_e.to, min_e.fr, min_e.cost, min_e.demand))
				self.caps[self.cnt] += min_e.demand
				remain -= min_e.demand
				endpoint = min_e.fr
				free.remove(min_e)
		self.cnt += 1

	def eval(self, routes=None):
		if routes is None:
			routes = self.routes
		tot_cost = 0
		for route in routes:
			_ = self.depot
			_dem = 0
			for e in route:
				tot_cost += DIS[_, e.fr] + e.cost
				_ = e.to
				_dem += e.demand
			tot_cost += DIS[_, self.depot]
		return tot_cost

	def __str__(self):
		sol = "s "
		for route in self.best_routes:
			sol += "0,"
			for e in route:
				sol += f"({e.fr + 1},{e.to + 1}),"
			sol += "0,"
		print(sol[:-1])
		print(f"q {int(self.best_cost)}")


def batch_solve(results, args, start_time):
	problems = [Problem(args.instance_file, i) for i in range(5)]
	for problem in problems:
		problem.path_scanning()
		problem.best_routes = deepcopy(problem.routes)
		problem.tot_cost = problem.eval()
		problem.best_cost = problem.tot_cost
	_iter = 0
	cfg = Config()
	LARGE_PROBLEM = problems[0].best_cost > 2000
	T = problems[0].best_cost * 0.05
	detr = 0.999
	EA_start_time = time.time()
	TIME_LIMIT = int(args.termination) - (EA_start_time - start_time)
	while time.time() - EA_start_time < TIME_LIMIT * 0.98:
		pid = _iter % 5
		CMO.perform_cmo(problems[pid], cfg, T)
		if problems[pid].tot_cost < problems[pid].best_cost:
			problems[pid].best_routes = deepcopy(problems[pid].routes)
			problems[pid].best_cost = problems[pid].tot_cost
		if _iter % (40 if LARGE_PROBLEM else 30) == 0:
			T *= detr
		_iter += 1
	best_pid = np.argmin([problem.tot_cost for problem in problems])
	results.put(problems[best_pid])


NUM_WORKERS = 2


if __name__ == "__main__":
	start_time = time.time()
	parser = argparse.ArgumentParser(description="CARP Solver")
	parser.add_argument("instance_file", help="CARP instance file")
	parser.add_argument("-t", "--termination", type=int, help="Termination condition")
	parser.add_argument("-s", "--seed", type=int, help="Random seed")
	args = parser.parse_args()
	random.seed(97)

	result_queue = multiprocessing.Queue()
	processes = []
	for i in range(NUM_WORKERS):
		p = multiprocessing.Process(target=batch_solve, args=(result_queue, args, start_time))
		processes.append(p)
		p.start()
	for p in processes:
		p.join()
	results = []
	while not result_queue.empty():
		result = result_queue.get()
		results.append(result)
	best_result = np.argmin([_.tot_cost for _ in results])
	results[best_result].__str__()
