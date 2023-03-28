import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from examples.components.shared.aml_smpc import AMLSMPC

class TestAMLSMPC(unittest.TestCase):
    def test_aml_smpc_basic(self):
        TEST_MSG = lambda recv: f"Message to {recv}"

        smpc1 = AMLSMPC()
        smpc2 = AMLSMPC()

        smpc1.add_remote_public_key(2, smpc2.get_public_key())
        smpc2.add_remote_public_key(1, smpc1.get_public_key())

        msg_2 = smpc1.encrypt(str.encode(TEST_MSG(2)), 2)
        msg_1 = smpc2.encrypt(str.encode(TEST_MSG(1)), 1)

        assert smpc1.decrypt(msg_1) == str.encode(TEST_MSG(1))
        assert smpc2.decrypt(msg_2) == str.encode(TEST_MSG(2))

    def test_aml_smpc_overwrite_key(self):
        TEST_MSG = lambda recv: f"Message to {recv}"

        smpc1 = AMLSMPC()
        smpc2 = AMLSMPC()
        smpc2_new = AMLSMPC()

        smpc1.add_remote_public_key(2, smpc2.get_public_key())
        smpc2.add_remote_public_key(1, smpc1.get_public_key())

        # Overwrite key
        smpc1.add_remote_public_key(2, smpc2_new.get_public_key(), overwrite=True)

        msg_2 = smpc1.encrypt(str.encode(TEST_MSG(2)), 2)
        msg_1 = smpc2.encrypt(str.encode(TEST_MSG(1)), 1)

        self.assertRaises(ValueError, smpc2.decrypt, msg_2)
        assert smpc1.decrypt(msg_1) == str.encode(TEST_MSG(1))
        assert smpc2_new.decrypt(msg_2) == str.encode(TEST_MSG(2))

    def test_aml_smpc_overwrite_fail(self):
        TEST_MSG = lambda recv: f"Message to {recv}"

        smpc1 = AMLSMPC()
        smpc2 = AMLSMPC()
        smpc2_new = AMLSMPC()

        smpc1.add_remote_public_key(2, smpc2.get_public_key())
        smpc2.add_remote_public_key(1, smpc1.get_public_key())

        # Overwrite key
        self.assertRaises(Exception, smpc1.add_remote_public_key, 2, smpc2_new.get_public_key())

        msg_2 = smpc1.encrypt(str.encode(TEST_MSG(2)), 2)
        msg_1 = smpc2.encrypt(str.encode(TEST_MSG(1)), 1)

        self.assertRaises(ValueError, smpc2_new.decrypt, msg_2)
        assert smpc1.decrypt(msg_1) == str.encode(TEST_MSG(1))
        assert smpc2.decrypt(msg_2) == str.encode(TEST_MSG(2))

if __name__ == '__main__':
    unittest.main()