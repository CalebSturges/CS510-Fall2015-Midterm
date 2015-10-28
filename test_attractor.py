from juliaset import JuliaSet
from nose import with_setup

###
# Test Suite for specified Attractor interface
#
# Run with the command: "nosetests attractor.py"
###

class TestRandomC:
    """Define an attractor with a test interface"""
    
    def setup(self):
        """Setup fixture is run before every test method separately"""
        self.at = Attractor(self.c, self.n)
        
    def test_s_value(self):
        """Test that the defalut s is correct"""
        assert self.at.s == 10.
    
    def test_p_value(self):
        """Test that the defalut p is correct"""
        assert self.at.p == 38.