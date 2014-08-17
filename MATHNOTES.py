
# because I'm always too dumb to remember:
def expDecay(t, tau):
    return np.exp(-t/tau)

def getTau(t, decay):
    return -t / np.log(decay)

def getHalflife(tau):
    return -tau * np.log(0.5)

def logsig(t):
    return 1.0 / (1.0 + np.exp(-t))


#  *** Parameters from Suri-Schultz-1999

# 100 ms per timestep.
# Future discount is 0.98 per timestep
# Past decay is 0.96 per timestamp

# half-life into the future is 3.43 s
##  (decay at 0.98 per 100ms => tau = 4.95)
## 5*tau = 24.75 s   (5*tau == 99.3% decay)

# half-life into the past is 1.7 s
##  (decay at 0.96 per 100ms  =>  tau = 2.44966)
## 5*tau = 12.25 s  (5*tau == 99.3% decay)


# The past must be greater than 1.7s, less than 12s
# The future must be greater than 3.5s, less than 24s


# 3*tau is 95% decay.
# +/- 3tau is:
#  -7.35s to +14.85s
# ... so let's go with 224 timesteps.  75 steps into the past and 149 steps into the future.

## New plan...  it helps to have a "lead-out", so that the final timestep in the LTM can
## be scrolled out through the echoic memory.  So:

#  - 75 steps into the past
#  - 150 steps into the future, 
#  - 75 steps of lead-out (just blank nothingness... all zeros)
#     .. for a total of 300 steps.


numNeurons = 10

neuronLabels = [
    'primaryRewardBlue',
    'primaryRewardRed',
    'actionA',
    'actionB',
    'cs0',
    'cs1',
    'cs2',
    'cs3',
    'cs4',
    'cs5']

assert len(neuronLabels) == numNeurons

# Primary reward (un-conditioned stimulus, UCS) -- the is the primary teaching
# signal coded into our DNA.  Food tastes good if we are hungry.  Water tastes
# good if we our thirsty.  This is hard-coded into our DNA.  This is the 
# primary teacher.
# CS - conditioned stimulus.  These are things that we learn.  After eating
# a ham sandwich a few times, we start to enjoy the sight and smell of a ham
# sandwich, even before tasting it.  (If we are hungry, anyway...)


# for the nn.w, row#0 is furthest into the future, row#223 is furthest into the past

#inputData = np.zeros(75, 10)
# for inputData, row#0 is the present, row#74 is furthest into the past



