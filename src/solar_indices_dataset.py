'''
Solar indices pytorch dataset

Reads in data from the Space Environment Technologies Solar Index datasets.

Indices_F10.csv:
    YYYY, DDD, JulianDay, F10, F81c, S10, S81c, M10, M81c, Y10, Y81c, Ssrc

    F81 is an 81 day smoothed F10.7 index
    S81 is an 81 day smoothed sunspot number index
    M81 is an 81 day smoothed Mg II index
    Y81 is an 81 day smoothed 10.7 cm solar radio flux index

    Simone says use F10 & S10 

'''

# TODO: Same as celestrak, linnea will copy over after she finished the celestrak dataset