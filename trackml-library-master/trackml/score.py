"""TrackML scoring metric"""

__authors__ = ['Sabrina Amrouche', 'David Rousseau', 'Moritz Kiehn',
               'Ilija Vukotic']

import numpy
import pandas

def _analyze_tracks(truth, submission, verbose=0):
    """Compute the majority particle, hit counts, and weight for each track.
    Technically, first create a dataframe with all hits and their associated ground truth and prediction,
    then sort by the prediction, and then the ground truth
    then for each predicted track, go over all hits and find the longest chain of hits which belong to the
    same ground truth particle, that will be the majority true particle for that predicted track, 
    and add its weight (which was inside the track) at the same time

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.

    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
        
        track_id = predicted track
        nhits = how many hits does predicted track contain
        major_particle_id = which particle from the truth is most often in the predicted track
        major_particle_nhits = how many hits in the truth file did this particle have
        major_nhits = how often is this true particle in the predicted track
        major_weight = what is the weight of all hits of this major particle WHICH lie in the track!
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    
    # particle_id is the truth // track_id is the prediction
    event = pandas.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)
    
    if verbose == 1:
        #print('printing out last 10 truth vs prediction')
        #print(event.tail(10))
        print('Truth: Num unique tracks: ', len(particles_nhits))
        #print('Truth: Num hits assigned to 0: ', (particles_nhits[0]))
        print('Truth: Num tracks with 1 hit: ', len(particles_nhits[particles_nhits==1]))
        print('Truth: Num max hits in 1 track (usually the 0 track): ', particles_nhits.max())

        
    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    # iterate over all combinations truth_track_id (=particle_id) and predicted_track_id (=track_id)
    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one, then overwrite the record
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weights = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track  # particle_id = truth
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pandas.DataFrame.from_records(tracks, columns=cols), event

def score_event(truth, submission, verbose=0):
    """Compute the TrackML event score for a single event.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """
    tracks, pd_event = _analyze_tracks(truth, submission, verbose)
        
    purity_rec = numpy.divide(tracks['major_nhits'], tracks['nhits'])  # to check there exists a majority particle in that track
    purity_maj = numpy.divide(tracks['major_nhits'], tracks['major_particle_nhits'])  # to check that this majority particle indeed was found in there
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    
    
    if verbose==1:
        #print(tracks.tail())
        print('Compare: num predicted tracks', len(tracks))
        print('Compare: num tracks, such that there exists 1 true majority particle with >50% of the hits', (0.5 < purity_rec).sum())
        print('Compare: num tracks, such that >50% of the majority particle hits are in the pred track', (0.5 < purity_maj).sum())
        print('Compare: num of tracks fulfilling both: ', good_track.sum())
    
    
        # from pd_event, take all rows, such that the particle_id is equal to the majority_particle_id of one good_track
        #print('\ntracks...')
        #print(tracks.tail())
        #print('\ngood_track...')
        #print(good_track.tail())
        #print('\ntake info of tracks, only for the good-track lines...')
        good_tracks = tracks['major_particle_id'][good_track]
        good_tracks = pandas.DataFrame(good_tracks)  # transform from series to dataframe with normal columnnames
        
           
        print('Compare: All predicted tracks - sum num hits', tracks['nhits'].sum())
        print('Compare: All predicted tracks - sum num hits of major particle', tracks['major_nhits'].sum())
        #print('Compare: All predicted tracks - Num major particle ids <> 0 in ground truth',len(tracks[tracks['major_particle_id'] != 0]))
        
        #print('Compare: All predicted tracks - Num unique major particle ids', len(tracks.groupby('major_particle_id')['major_particle_id'].nunique()))
        
        p = pd_event[pd_event['particle_id'].isin(good_tracks['major_particle_id'])]
        s = p['weight'].sum()
        print('Good tracks: weight sum of hits of the maj particles of tracks fulfilling both (=if u had them fully):', round(s,4))
         
        
        t = tracks[tracks['major_particle_id'].isin(good_tracks['major_particle_id'])]
        t = t.drop_duplicates(subset='major_particle_id', keep="last")
        #print('t')
        #print(t)
        
        print('Good tracks: All predicted tracks - sum num hits of major particle in truth ', t['major_particle_nhits'].sum())
        
    
    return tracks['major_weight'][good_track].sum()
