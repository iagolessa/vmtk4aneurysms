# Copyright (C) 2022, Iago L. de Oliveira

# vmtk4aneurysms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Collection of general tools."""

import sys

def FlattenDict(
        pyobj,
        keystring=''
    ):

    if type(pyobj) == dict:
        keystring = keystring + '_' if keystring else keystring

        for k in pyobj:
            yield from FlattenDict(pyobj[k], keystring + str(k))

    else:
        yield keystring, pyobj
