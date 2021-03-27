
class iEEG(object):
	"""docstring for iEEG"""
	def __init__(self, label, classes, epoch=None):
		super(iEEG, self).__init__()
		self.epoch = epoch
		self.label = label
		self.classes = classes

	def epoch(self, epoch):
		self.epoch = epoch

	def X(self, X):
		self.X = X

	def hilb(self,lfr,hfr):
		return self.epoch.copy().filter(l_freq=f,h_freq=f+10,n_jobs=-1,fir_design='firwin',picks='all').apply_hilbert(n_jobs=-1,envelope=True)

	def get_X(self, plage=[(a,a+10) for a in range(10,120,10)], f_bin=10, tranformation="Hilbert"):
		"""
			Fonction pour obtenir le vecteur X à partir de la classe époch
		"""
		stacked_X = []

		#Itération sur les plage de fréquences qui nous intéressent
		for (lfr, hfr) in plage:
			

			"""
				1) Produit une liste de signaux transformé par Hilbert puis normalisé au sein des contacts
				2) Concatène les signaux des contacts normalisé
				3) Moyenne les signaux pour n'en avoir qu'un par contact

				Shape epoch.get_data() : m_epochs, n_contacts, o_millisecondes
			"""
			out = np.concatenate([sts.zscore(hilb(f,f+f_bin).get_data(),axis=2) for f in range(lfr,hfr,f_bin)],axis=2).mean(axis=2)
			ep_out = mne.EpochsArray(out,
									info= mne.create_info( ch_names=self.epoch.ch_names,
														   ch_types=['eeg' for x in range(len(self.epoch.ch_names))],
														   sfreq=self.epoch.info["sfreq"]),
									tmin=self.epoch.tmin)

			del out
			"""
				Baseline Normalisation

				1) Construire le vecteur prè-stimulus
				2) Construire le vecteur post-stimulus
				3) Soustraire la baseline
				4) Renvoyer le vecteur
			"""

			#Pre-stimulus -100 ms jusqu'à 0 ms
			X_base = ep_out.copy().crop(tmin=-0.1,tmax=0.0).get_data()
			X_ep = ep_out.copy().crop(tmin,tmax).get_data()

			binnage = lambda a,b : np.apply_along_axis(lambda c: sts.binned_statistic(range(len(c)),c,bins=b)[0],axis=2,arr=a)

			if X_bin !=0 : 
				# Créer N morceaux du signaux, qui sont moyennés
				X = np.nan_to_num(binnage(X_ep,X_bin) - binnage(X_base,X_bin),nan=0.0,posinf=0.0,neginf=0.0)
    			X = X.reshape((X.shape[0],np.prod(X.shape[1:])))
    		else:
    			X = np.concatenate([X_0b,X_00], axis=2)

    		stacked_X.append(X)

    		del X




		def X_y(ep_hilb,tmin,tmax,cond=0,bin=1):
  if cond == 0:
    X = get_X(ep_hilb,tmin,tmax,bin=bin)
    y = np.array(ep_hilb.metadata.loc[:,"error"]) 
  else:
    X = get_X(ep_hilb[cond],tmin,tmax,bin=bin)
    y = np.array(ep_hilb[cond].metadata.loc[:,"error"]) 
  return X,y