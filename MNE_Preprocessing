import os
import mne
from mne import io
from mne.report import Report
import numpy as np


#----------------------Data description---------------------------#
###################################################################
path =  "path"
name = "name"
file_name = path + name

raw = io.read_raw_fif(file_name, preload = True) 
print(raw)
raw.info
print(raw.ch_names)
nchans = raw.info['nchan']
print(nchans)

raw.plot(n_channels=10, scalings = 'auto')

# sensor position
raw.plot_sensors(kind='topomap',show_names=True);

# psd
raw.plot_psd (fmin=1., fmax = 45., tmax = 50., average = False)
raw.plot_psd_topo(fig_facecolor='w', axis_facecolor='w', color='k'); # topomap for every chan
raw.plot_psd(area_mode='std', average=True);

# chosen channels
picks = mne.pick_channels(raw.ch_names, include=['Pz','CPz','Cz','FCz','Fz','C1','C2','FC1','FC2','CP1','CP2'])
#raw(picks = picks).plot()

#----------------------Data manipulation---------------------------#
###################################################################

#filter
raw=raw.copy().filter(0.1, 40., fir_design='firwin')

raw.notch_filter(np.arange(50,150,200), n_jobs=1, fir_design='firwin')

#events
events = mne.find_events(raw, stim_channel='STI101',shortest_event=1)
print (events)
events_id = {'visual1': 37, 'visual2': 77, 'audio1':117,'audio2': 157}
mne.viz.plot_events(events, raw.info['sfreq']) 

#epochs 
epochs = mne.Epochs(raw, events , event_id=events_id, tmin=-0.2, tmax=0.5, baseline=(None, 0), reject_by_annotation = True , verbose=True, preload=True) #reject = //, 
print(epochs)
epochs_visual_abd_audio = mne.Epochs(raw, events, event_id=events_id, tmin=-0.2, tmax=0.5) #, reject=reject
epochs_visual_abd_audio.plot()
epochs.plot_image(combine='mean')

picks = mne.pick_channels(raw.ch_names, include=['Fz','FCz','Cz','CPz','Pz'])
epochs.plot_image(picks = picks, combine='mean', scalings=scalings, units=units)


#----------------------ICA----------------------------------------#
###################################################################
ica = mne.preprocessing.ICA(n_components=40, random_state=97, max_iter=800,method = 'fastica')

# ica = mne.preprocessing.read_ica('path/namefile.fif')
ica.fit(raw)
print(ica)

ica.plot_sources(raw)
fmin, fmax = 1, 100 
ica.plot_components(inst = raw,psd_args={'fmax': 50}) #topomap_args={'ch_type‘:mag}

# ica.exclude = [0, 18]
raw_clean = ica.apply(raw) # apply the changes

ica_name_out = 'Subject1_ICA'
data_clear_name = 'Subject1_clear'
ica.save(ica_name_out) #you could save your components
raw_clean.save(data_clear_name) 

#---------------------Autorejection-------------------------------#
###################################################################
from autoreject import AutoReject, compute_thresholds 
from autoreject import get_rejection_threshold 

# global rejection threshold
reject = get_rejection_threshold(epochs)
print(reject)
epochs.drop_bad(reject=reject)
epochs.average().plot() 

# local rejection threshold
this_epoch = epochs['visual1']
exclude = []  # XXX
picks = mne.pick_types(epochs.info, meg=True, eeg=False, stim=False, eog=False, exclude=exclude)

ar = AutoReject(picks=picks, random_state=42, n_jobs=1, verbose='tqdm')
epochs_ar, reject_log = ar.fit_transform(this_epoch, return_log=True)

reject_log.plot_epochs(this_epoch, scalings='auto')
epochs_ar.plot(scalings='auto')

epochs_ar.plot_drop_log() #dropp epochs

#---------------------Report-------------------------------#
###################################################################
name = 'Subject1_report'
report = Report(image_format='png',raw_psd=False,title=name,subject=name)

#generate ICA
fig =ica.plot_components(inst = raw)
report.add_htmls_to_section("Filter parameters: 0.1-50 Hz, fir_design='firwin'; ICA parameters: n_components=40, random_state=97, max_iter=800, method = 'fastica'",captions= "Description") 

report.add_slider_to_section(fig, title="ICA",section ="ICA_components",image_format='png') 

# generate images with properties
n = list(range(0, 40))
fig_2 = ica.plot_properties(raw, picks=n, psd_args={'fmax': 50})
report.add_slider_to_section(fig_2, title="ICA",section ="ICA_components_properties",image_format='png') 
report.save('/Users/n/PycharmProjects/application/'+name+'.html', overwrite=True)
