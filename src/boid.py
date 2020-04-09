import numpy as np
from mesa import Agent



class Boid(Agent):
    '''
    A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents.
        - Separation: avoiding getting too close to any other agent.
        - Alignment: try to fly in the same direction as the neighbors.

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and velocity (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    '''
    def __init__(self, unique_id, model, pos, speed, velocity, vision,
            separation, initial_status, infection_time, cohere=0.025, separate=0.25, match=0.04):
        '''
        Create a new Boid flocker agent.

        Args:
            unique_id: Unique agent identifyer.
            pos: Starting position
            speed: Distance to move per step.
            heading: numpy vector for the Boid's direction of movement.
            vision: Radius to look around for nearby Boids.
            separation: Minimum distance to maintain from other Boids.
            cohere: the relative importance of matching neighbors' positions
            separate: the relative importance of avoiding close neighbors
            match: the relative importance of matching neighbors' headings

        '''
        # 追加
        # state:患者の状態
        # susceptible:感染しうる状態
        # infected:感染
        # recovered:抗体保持
        # removed:離脱
        
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.velocity = velocity
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.status = initial_status
        self.infection_time = 0
        self.motality = 0.1         # 死亡率
        self.infection_rate = 0.5   # 感染率

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of mass of the local neighbors.
        '''
        cohere = np.zeros(2)
        if neighbors:
            for neighbor in neighbors:
                cohere += self.model.space.get_heading(self.pos, neighbor.pos)
            cohere /= len(neighbors)
        return cohere

    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist.
        '''
        me = self.pos
        them = (n.pos for n in neighbors)
        separation_vector = np.zeros(2)
        for other in them:
            if self.model.space.get_distance(me, other) < self.separation:
                separation_vector -= self.model.space.get_heading(me, other)
        return separation_vector

    def match_heading(self, neighbors):
        '''
        Return a vector of the neighbors' average heading.
        '''
        match_vector = np.zeros(2)
        if neighbors:
            for neighbor in neighbors:
                match_vector += neighbor.velocity
            match_vector /= len(neighbors)
        return match_vector

    def step(self):
        '''
        Get the Boid's neighbors, compute the new vector, and move accordingly.
        '''

        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        self.velocity += (self.cohere(neighbors) * self.cohere_factor +
                          self.separate(neighbors) * self.separate_factor +
                          self.match_heading(neighbors) * self.match_factor) / 2
        self.velocity /= np.linalg.norm(self.velocity)
        new_pos = self.pos + self.velocity * self.speed
        self.infection_recover(neighbors)
        self.model.space.move_agent(self, new_pos)
  
    def infection_recover(self, neighbors):
        '''
        近傍のエージェントから感染する.
        すでに感染している場合は一定確率で回復or死亡
        '''
        if neighbors:
            if(self.status == "susceptible"):
                for neighbor in neighbors:
                    if(neighbor.status == "infected"):
                        if(self.random.random() < self.infection_rate):
                            self.state = "infected"
        if(self.status == "infected"):
            if(self.infection_time > 10):
                if(self.random.random() < self.motality):
                    self.state = "removed"
                else:
                    self.state = "recovered"
            self.infection_time += 1
