import main_functions as f
import numpy as np

# test dynamics-----------------------------------------------------------------
patterns = f.generate_patterns(80,1000)

f.test_main(patterns,200,'dynamics','hebbian',20,0)

# test dynamics_async-----------------------------------------------------------------

patterns = f.generate_patterns(80,1000) 

f.test_main(patterns,200,'dynamics_async','hebbian',20000,3000)

#test of storkey dynamics -------------------------------------------------------------------------------------------------------

patterns = f.generate_patterns(80,1000)

f.test_main(patterns,200,'dynamics','storkey',20,0)

#test of storkey dynamics_async -------------------------------------------------------------------------------------------------------

patterns = f.generate_patterns(80,1000) 

f.test_main(patterns,200,'dynamics_async','storkey',20000,3000)
