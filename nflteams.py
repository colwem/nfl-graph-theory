class NFLTeams:



    class Team(object):

        """Docstring for Team. """

        def __init__(self, name):
            """@todo: to be defined1.

            :name: full name of team

            """
            self.name = name

        @property
        def PFR(self):
            return NFLTeams._name_map[self.name]['PFR']

        @property
        def flair(self):
            return NFLTeams._name_map[self.name]['flair']

        def __repr__(self):
            return "<NFLTeams, name: " + self.name + ">"

    def __init__(self):
        self._teams = {}

    @property
    def teams(self):
        return self._teams.values()

    def get_team(self, name):
        if not self._name_map.has_key(name):
            raise Exception("'" + name + "' is not a valid team name")
        if self._teams.has_key(name):
            return self._teams[name]
        else:
            self._teams[name] = self.Team(name)
            return self._teams[name]

    def add_attrs(self, name, vals, ordered_by=None):
        if not type(vals) is dict:
            if ordered_by:
                things = zip(ordered_by, vals)
            else:
                things = zip(self._teams.values(), vals)
        else:
            things = vals.items()
        for k,v in things:
            k.__setattr__(name, v)

    def get_attrs(self, name, ordered_by=None):
        """@todo: Docstring for get_attrs.

        :attr: attribute
        :ordered_by: list of Teams in the order that we need the attributes
        :returns: List containing the attribute

        """
        if not ordered_by:
            ordered_by = self._teams.values()
        return [t.__getattribute__(name) for t in ordered_by]

    def set_attrs(self, name, vals, ordered_by=None):
        if not type(vals) is dict:
            if ordered_by:
                things = zip(ordered_by, vals)
            else:
                things = zip(self._teams.values(), vals)
        else:
            things = vals.items()
        for k,v in things:
            k.__setattr__(name, v)

    def get_table(self, columns=None, sorted_by=None):
        table = []
        for t in self._teams.values():
            row = []
            for c in columns:
                row.append(t.__getattribute__(c))
            table.append(row)
        if sorted_by:
            table.sort(key=lambda x: x[sorted_by])
        return table


    _name_map = {
        'Seattle Seahawks'     : {'PFR': 'SEA', 'flair': "[](/SEA)"},
        'New York Jets'        : {'PFR': 'NYJ', 'flair': "[](/NYJ)"},
        'Houston Texans'       : {'PFR': 'HOU', 'flair': "[](/HOU)"},
        'Carolina Panthers'    : {'PFR': 'CAR', 'flair': "[](/CAR)"},
        'Atlanta Falcons'      : {'PFR': 'ATL', 'flair': "[](/ATL)"},
        'Pittsburgh Steelers'  : {'PFR': 'PIT', 'flair': "[](/PIT)"},
        'Minnesota Vikings'    : {'PFR': 'MIN', 'flair': "[](/MIN)"},
        'Buffalo Bills'        : {'PFR': 'BUF', 'flair': "[](/BUF)"},
        'Tennessee Titans'     : {'PFR': 'TEN', 'flair': "[](/TEN)"},
        'Denver Broncos'       : {'PFR': 'DEN', 'flair': "[](/DEN)"},
        'Miami Dolphins'       : {'PFR': 'MIA', 'flair': "[](/MIA)"},
        'Cincinnati Bengals'   : {'PFR': 'CIN', 'flair': "[](/CIN)"},
        'San Francisco 49ers'  : {'PFR': 'SFO', 'flair': "[](/SF)"},
        'Philadelphia Eagles'  : {'PFR': 'PHI', 'flair': "[](/PHI)"},
        'Detroit Lions'        : {'PFR': 'DET', 'flair': "[](/DET)"},
        'Arizona Cardinals'    : {'PFR': 'ARZ', 'flair': "[](/ARI)"},
        'Green Bay Packers'    : {'PFR': 'GNB', 'flair': "[](/GB)"},
        'Oakland Raiders'      : {'PFR': 'OAK', 'flair': "[](/OAK)"},
        'Washington Redskins'  : {'PFR': 'WAS', 'flair': "[](/WAS)"},
        'Tampa Bay Buccaneers' : {'PFR': 'TAM', 'flair': "[](/TB)"},
        'New Orleans Saints'   : {'PFR': 'NOR', 'flair': "[](/NO)"},
        'Cleveland Browns'     : {'PFR': 'CLE', 'flair': "[](/CLE)"},
        'St. Louis Rams'       : {'PFR': 'STL', 'flair': "[](/STL)"},
        'Chicago Bears'        : {'PFR': 'CHI', 'flair': "[](/CHI)"},
        'Kansas City Chiefs'   : {'PFR': 'KAN', 'flair': "[](/KC)"},
        'Indianapolis Colts'   : {'PFR': 'IND', 'flair': "[](/IND)"},
        'New England Patriots' : {'PFR': 'NWE', 'flair': "[](/NE)"},
        'Baltimore Ravens'     : {'PFR': 'BAL', 'flair': "[](/BAL)"},
        'Dallas Cowboys'       : {'PFR': 'DAL', 'flair': "[](/DAL)"},
        'Jacksonville Jaguars' : {'PFR': 'JAX', 'flair': "[](/JAC)"},
        'New York Giants'      : {'PFR': 'NYG', 'flair': "[](/NYG)"},
        'San Diego Chargers'   : {'PFR': 'SDG', 'flair': "[](/SD)"},
    }
