import picobot as p
import random
import copy
import numpy as np
from statistics import mean
import sys
import time
import matplotlib.pyplot as plt

# ASCII_CHARS_COUNT = 256
# AVAILABLE_OPS = [">", # Increment the pointer.
#  				 "<", # Decrement the pointer.
#  				 "+", # Increment the byte at the pointer.
#  				 "-", # Decrement the byte at the pointer.
#  				 ".", # Output the byte at the pointer.
#  				 "[", # Jump forward past the matching ] if the byte at the pointer is zero.
#  				 "]"] # Jump backward to the matching [ unless the byte at the pointer is zero.
# 				 #"," #Input a byte and store it in the byte at the pointer. (Since we don't want any inputs, let's skip it as for now)

LAYOUT_FILE = 'layout.txt'
MAX_PROGRAM_SIZE = 4
MAX_STATE_NUMBER = 4

POPULATION = 100
MUTATION_RATE = 0.115
MAX_MUTATION_ATTEMPTS = 30
SELECTION_RATE = 0.2
TOP_PERFORMERS_COUNT = int(POPULATION * SELECTION_RATE)
OUTPUT_DIR = "./output/"
PROGRAM_DIR = "./programs/"

POSSIBLE_DIRECTIONS = [p.NORTH, p.SOUTH, p.EAST, p.WEST]
POSSIBLE_STATES = list(range(0, MAX_STATE_NUMBER + 1))


def generate_mask(direction, x:bool) -> str:
	if not x:
		if direction == p.NORTH:
			return "N***"
		elif direction == p.SOUTH:
			return "***S"
		elif direction == p.EAST:
			return "*E**"
		elif direction == p.WEST:
			return "**W*"
	if x:
		if direction == p.NORTH:
			return "x***"
		elif direction == p.SOUTH:
			return "***x"
		elif direction == p.EAST:
			return "*x**"
		elif direction == p.WEST:
			return "**x*"
	raise NameError("Could not generate mask")

POSSIBLE_SURROUNDINGS = [generate_mask(d, True) for d in POSSIBLE_DIRECTIONS] + [generate_mask(d, False) for d in POSSIBLE_DIRECTIONS]

class Meta_Program_Line():
	def __init__(self, state1, direction1, x:bool, direction2, state2):
		
		assert direction1 in [p.NORTH, p.SOUTH, p.EAST, p.WEST] and direction2 in [p.NORTH, p.SOUTH, p.EAST, p.WEST]
		assert state1 in list(range(0, MAX_STATE_NUMBER + 1)) and state2 in list(range(0, MAX_STATE_NUMBER + 1))
		self.state1 = state1
		self.surroundings = generate_mask(direction1, x)
		self.direction = direction2
		self.state2 = state2

	def toString(self):
		
		s = ""
		s += str(self.state1)
		s += ' '
		s += str(self.surroundings)
		s += ' -> '
		s += str(self.direction)
		s += ' '
		s += str(self.state2)
		return s

class Meta_Program():
	def __init__(self, lines:list, id:int):
		for line in lines:
			assert isinstance(line, Meta_Program_Line)
		self.lines = lines
		assert isinstance(id, int)
		self.id = id
		self.filename = "Program" + str(id) + ".pico"

	def toString(self) -> str:
		s = ''
		for line in self.lines:
			if len(s) != 0:
				s += '\n'
			s += line.toString()
		return s

	def __repr__(self):
		return f"#PROGRAM {self.id}\n" + self.toString() + "\n"

class GeneticEvolutionMetaProgrammer():

	generation = 0
	population = []
	max_fitness_score = 0
	start_time = None
	best_fitness_scores = []

	def __init__(self):
		self.start_time = time.time()
		self.max_fitness_score = 100
		print("Start")
		self.genetic_evolution()

	def genetic_evolution(self):
		valid_population = []
		while True:
			print("\ngeneration: " + str(self.generation) + ", population: " + str(len(self.population)) + ", mutation_rate: " + str(MUTATION_RATE))
			
			# 1. Selection
			chromosomes, elite = self.select_elite(valid_population)
			print("ELITE", elite, "len", len(elite))
			for e in elite:
				print(e[0].id, e[1])
			assert all([isinstance(x, tuple) for x in elite])
			# 2. Crossover (Roulette selection)
			pairs = self.generate_pairs(elite)
			print("-----------PAIRS-------------", pairs, "len", len(pairs))
			selected_offsprings = []
			for pair in pairs:
				offsprings = self.crossover(pair[0][0], pair[1][0])
				selected_offsprings.append(offsprings[random.randint(0, 1)])
			print("SELCTED OFFSPRINGS", selected_offsprings, "len", len(selected_offsprings))
			# 3. Mutation
			mutated_population = self.mutation(selected_offsprings)
			print("MUTATED POPULATION", mutated_population, "len", len(mutated_population))

			# 4. Validation (We don't want syntactically incorrect programs)
			valid_population = []
			for chromosome in mutated_population:
				if brainfuck.evaluate(chromosome) is not None:
					valid_population.append(chromosome)
			print("propagated to next generation: " + str(len(valid_population)))
			self.population = self.generate_population(valid_population)
			self.generation += 1

	def generate_population(self, population):
		
		while len(population) < POPULATION:
			lines = []
			# Hardcode the first line to open with state 0
			lines.append(Meta_Program_Line(0, \
						random.choice(POSSIBLE_DIRECTIONS), random.choice([True, False]),\
						random.choice(POSSIBLE_DIRECTIONS), random.choice(POSSIBLE_STATES)))
			for i in range(random.randint(0, MAX_PROGRAM_SIZE-1)):
				lines.append(Meta_Program_Line(random.choice(POSSIBLE_STATES), \
						random.choice(POSSIBLE_DIRECTIONS), random.choice([True, False]),\
						random.choice(POSSIBLE_DIRECTIONS), random.choice(POSSIBLE_STATES)))
			program = Meta_Program(lines, "Program")
			
			# length = random.randint(PROGRAM_LENGTH_LOWER_BOUND, PROGRAM_LENGTH_UPPER_BOUND)
			# length = random.randint(PROGRAM_LENGTH_LOWER_BOUND, PROGRAM_LENGTH_UPPER_BOUND)
			# for i in range(0, length):
			# 	chromosome += random.choice(AVAILABLE_OPS)
		return population


	def select_elite(self, population):
		try:
			all_numbers = [int(p.id) for p in population]
		except ValueError:
			raise NameError("Currupt population")
		scores_for_chromosomes = []
		cntr = 0
		while cntr < POPULATION:
			if cntr in all_numbers:
				cntr += 1
				continue
			# Hardcode the first line to open with state 0
			lines = []
			lines.append(Meta_Program_Line(0, \
						random.choice(POSSIBLE_DIRECTIONS), random.choice([True, False]),\
						random.choice(POSSIBLE_DIRECTIONS), random.choice(POSSIBLE_STATES)))
			for i in range(random.randint(0, MAX_PROGRAM_SIZE-1)):
				lines.append(Meta_Program_Line(random.choice(POSSIBLE_STATES), \
						random.choice(POSSIBLE_DIRECTIONS), random.choice([True, False]),\
						random.choice(POSSIBLE_DIRECTIONS), random.choice(POSSIBLE_STATES)))
			program_filename = "Program" + str(cntr) + ".pico"
			program = Meta_Program(lines, cntr)
			print("Testing", program.filename)
			with open(program_filename, "w") as f:
				f.write(program.toString())
				f.close()
			try:
				result = p.evaluate(program_filename)
			except Exception or NameError:
				del program
				continue
			else:
				cntr += 1
			score = result
			scores_for_chromosomes.append((program, score))
			
		
		# for i in range(0, len(population)):
		# 	chromosome = population[i]
		# 	result = p.evaluate(chromosome)
		# 	print(result)
		# 	if not result: continue
		# 	score = result
			# if score == self.max_fitness_score:
			# 	current_time = time.time()
			# 	print("\nFOUND SOLUTION: " + chromosome + " for: " + repr(self.target) +  " in: " + str(int((current_time-self.start_time)/60)) + " minutes")
			# 	self.best_fitness_scores.append(self.max_fitness_score)
			# 	self.update_fitness_plot()
			# 	exit()
		chromosomes = [x[0] for x in scores_for_chromosomes]
		scores_for_chromosomes.sort(key=lambda x: x[1])
		scores = [x[1] for x in scores_for_chromosomes]
		print("population: " + "(min: " + str(min(scores)) + ", avg: " + str(mean(scores)) + ", max: " + str(max(scores)) + ")")

		top_performers = scores_for_chromosomes[-TOP_PERFORMERS_COUNT:]
		top_scores = [x[1] for x in top_performers]
		print("elite " + str(round(1.0-SELECTION_RATE, 2)) + ": " + "(min: " + str(min(top_scores)) + ", avg: " + str(mean(top_scores)) + ", max: " + str(max(top_scores)) + ")")
		
		chromosome = top_performers[-1][0]
		result = p.evaluate(chromosome.filename)
		 
		best_fitness_score = result
		print("best: " + chromosome.filename + str(chromosome) + ", result: " + repr(result) + ", score: " + str(best_fitness_score) + "/" + str(self.max_fitness_score))
		self.best_fitness_scores.append(best_fitness_score)
		# self.update_fitness_plot()
		
		return chromosomes, top_performers

	def generate_pairs(self, parents):
		normalized_parents = self.softmax([x[1] for x in parents])
		total_parents_score = sum(normalized_parents)
		pairs = []
		while len(pairs) < POPULATION:
			pair = self.pair(parents, normalized_parents, total_parents_score)
			if len(pair) == 2 and pair[0] is not None and pair[1] is not None:
				pairs.append(pair)
		return pairs

	def pair(self, parents, normalized_parents, total_parents_score):
		pick_parent_a = random.uniform(0, total_parents_score)
		pick_parent_b = random.uniform(0, total_parents_score)
		return [self.roulette_selection(parents, normalized_parents, pick_parent_a), self.roulette_selection(parents, normalized_parents, pick_parent_b)]

	def roulette_selection(self, parents, normalized_parents, pick):
		current = 0.0
		for i in range(0, len(parents)):
			current += normalized_parents[i]
			if current > pick:
				return parents[i]

	def crossover(self, x:Meta_Program, y:Meta_Program):
		offspring_x = x
		offspring_y = y
		# length = min(len(x), len(y))
		for i in range(0, 4):
			# 1: State, 2: Surroundings, 3: Direction, 4:End State
			if random.choice([True, False]):
				line_x = offspring_x.lines[random.randint(0, len(offspring_x.lines)-1)]
				line_y = offspring_y.lines[random.randint(0, len(offspring_y.lines)-1)]
				if i == 0:
					s1 = line_x.state1
					s2 = line_y.state1
					line_y.state1 = s1
					line_x.state1 = s2
				if i == 1:
					s1 = line_x.surroundings
					s2 = line_y.surroundings
					line_y.surroundings = s1
					line_x.surroundings = s2
				if i == 2:
					s1 = line_x.direction
					s2 = line_y.direction
					line_y.direction = s1
					line_x.direction = s2
				if i == 3:
					s1 = line_x.state2
					s2 = line_y.state2
					line_y.state2 = s1
					line_x.state2 = s2
					
		return offspring_x, offspring_y

	def crossover_at_index(self, x, y, i):
		x_at_i = self.get_value_at_index(x, i)
		y_at_i = self.get_value_at_index(y, i)
		x = self.set_value_at_index(x, y_at_i, i)
		y = self.set_value_at_index(y, x_at_i, i)
		return x, y

	def mutation(self, selected_offsprings):
		offsprings = []
		for offspring in selected_offsprings:
			print("Mutating", offspring.filename, end='')
			valid = False
			mutation_attempts = 0
			while not valid and mutation_attempts < MAX_MUTATION_ATTEMPTS:
				offspring_mutation = copy.deepcopy(offspring)
				if np.random.choice([True, False], p=[MUTATION_RATE, 1-MUTATION_RATE]):
					# only type of mutation that makes sense for this type of program is replace
					location = np.random.randint(0, 3)
					if location == 0:
						offspring_mutation.state1 = np.random.choice(POSSIBLE_STATES)
					if location == 1:
						offspring_mutation.surroundings = np.random.choice(POSSIBLE_SURROUNDINGS)
					if location == 2:
						offspring_mutation.direction = np.random.choice(POSSIBLE_DIRECTIONS)
					if location == 3:
						offspring_mutation.state2 = np.random.choice(POSSIBLE_STATES)
				with open(offspring_mutation.filename, "w") as f:
					f.write(offspring_mutation.toString())
					f.close()
				try:
					p.evaluate(offspring_mutation.filename)
				except Exception or NameError:
					mutation_attempts += 1
					continue
				else:
					valid = True
					offsprings.append(offspring_mutation)
			print("# of attempts", mutation_attempts)
				
		return offsprings

	def update_fitness_plot(self):
		plt.plot(self.best_fitness_scores, label="best_fitness")
		plt.plot([self.max_fitness_score for _ in range(0, len(self.best_fitness_scores))], label="max_fitness (" + str(self.max_fitness_score) + ")")
		plt.legend(loc='best')
		plt.title("Target: " + repr(self.target))
		plt.xlabel("Generation")
		plt.ylabel("Fitness")
		plt.savefig(OUTPUT_DIR + repr(self.target) + ".png", bbox_inches="tight")
		plt.close()

	def softmax(self, x):
		y = np.exp(x - np.max(x))
		return y / y.sum()

	def get_value_at_index(self, string, i):
		try:
			return string[i]
		except IndexError:
			return None

	def set_value_at_index(self, string, value, i):
		if i > len(string):
			return string
		elif value is not None:
			return string[:i] + value + string[i+1:]
		else:
			return string

if __name__ == "__main__":
	GeneticEvolutionMetaProgrammer()
	
