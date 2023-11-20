import random
import config
import numpy as np


class Mixins:
    def checkType(self, obj):
        if not isinstance(obj, Particle):
            raise TypeError("Can only opererate on Particle type")


class SwapSequence(Mixins, object):
    def __init__(self, sequence: list[tuple[int, int]]):
        self.sequence = sequence

    def __add__(self, seq):
        selfSeqCopy = self.sequence.copy()
        selfSeqCopy.extend(seq.sequence)
        return SwapSequence(selfSeqCopy)


class ParticlePosition:
    def __init__(self, position: list[tuple[int, int, int]]):
        self.location = position

    def __sub__(self, particlePos):
        sequence = []
        for idx, item in enumerate(particlePos.location):
            index = particlePos.location.index(item)
            sequence.append((idx, index))
        swapSequence = SwapSequence(sequence)
        return swapSequence

    def __add__(self, seq: SwapSequence):
        posCopy = self.location.copy()
        for operator in seq.sequence:
            temp = posCopy[operator[0]]
            posCopy[operator[0]] = posCopy[operator[1]]
            posCopy[operator[1]] = temp
        newParticlePosition = ParticlePosition(posCopy)
        return newParticlePosition


class Particle(Mixins, object):
    def __init__(self, position: list[tuple[int, int, int]], velocity: SwapSequence):
        self.position = ParticlePosition(position)
        self.velocity = velocity
        self.bestPosition = self.position
        # Initialize to negative infinity for maximization
        self.bestFitness = float('-inf')


class PSO:
    bestPosition: list[tuple[int, int, int]] = None
    bestFitness = float('-inf')
    bestTime = None

    def __init__(self, numParticles=20, numDomain=200, numIterations=30,
                 cognitiveWeight=0.5, socialWeight=0.5,
                 randomInstance=random.SystemRandom()):
        self.numParticles = numParticles
        self.numDomain = numDomain
        self.numIterations = numIterations
        self.cognitiveWeight = cognitiveWeight
        self.socialWeight = socialWeight
        self.randomInstance = randomInstance

    def createDomain(self) -> list[list[tuple[int, int, int]]]:
        domain = []
        for _ in range(self.numDomain):
            while True:
                searchDomain = self.createSearchDomain()
                if searchDomain not in domain:
                    domain.append(searchDomain)
                    break
        return domain

    def createPopulation(self, searchDomain: list[tuple[int, int, int]]) -> list[Particle]:
        particles = []
        existingDomain = []
        for _ in range(self.numParticles):
            searchDomainCopy = searchDomain.copy()
            while searchDomainCopy in existingDomain:
                self.randomInstance.shuffle(searchDomainCopy)
            velocity = SwapSequence(
                [tuple(self.randomInstance.sample(range(0, 4), 2))])
            existingDomain.append(searchDomainCopy)
            particle = Particle(searchDomainCopy, velocity)
            particles.append(particle)
        return particles

    def createSearchDomain(self) -> list[tuple[int, int, int]]:
        particleSolution = []
        for item in config.ORDER:
            chosenItem = ''
            while True:
                itemLocations = config.item_location_mapping[item]
                chosenItem = itemLocations[np.random.choice(
                    len(itemLocations))]
                if chosenItem not in particleSolution:
                    break
            particleSolution.append(chosenItem)
        self.randomInstance.shuffle(particleSolution)
        return particleSolution

    def fitnessFunc(self, solution) -> float:
        return 1/config.total_t(solution)

    # V(t+1) = V(t) + c1*(P(i)(best) - X(i)) + c2*(P(g)(best) - X(i))

    def updateVelocity(self, particle: Particle, bestPos: ParticlePosition):
        velocity = particle.velocity
        if self.randomInstance.choices([0, 1], [1 - self.cognitiveWeight, self.cognitiveWeight])[0]:
            velocity += (particle.bestPosition - particle.position)
        if self.randomInstance.choices([0, 1], [1 - self.socialWeight, self.socialWeight])[0]:
            velocity += (bestPos - particle.position)

        return velocity

    def run(self):
        domains = self.createDomain()
        for domain in domains:
            particles = self.createPopulation(domain)

            domainBestFitness = float('-inf')
            domainBestPosition = None
            for _ in range(self.numIterations):
                for particle in particles:
                    fitness = self.fitnessFunc(particle.position.location)
                    if fitness > particle.bestFitness:
                        particle.bestFitness = fitness
                        particle.bestPosition = particle.position

                    if fitness > domainBestFitness:
                        domainBestFitness = fitness
                        domainBestPosition = particle.position

                    if domainBestFitness > self.bestFitness:
                        self.bestFitness = domainBestFitness
                        self.bestPosition = domainBestPosition
                        self.bestTime = config.total_t(
                            self.bestPosition.location)

                    particle.velocity = self.updateVelocity(
                        particle, domainBestPosition)
                    particle.position += particle.velocity

        return self.bestPosition.location, self.bestTime


if __name__ == "__main__":
    pso = PSO()

    best_position, best_fitness = pso.run()

    print("Best Position:", best_position)
    print("Best Fitness:", best_fitness)
