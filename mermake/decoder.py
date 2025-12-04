import time,sys
import os,sys,numpy as np

def get_intersV2(self,nmin_bits=3,dinstance_th=2,enforce_color=True,enforce_set=None,redo=False):
	"""Get an initial intersection of points and save in self.res"""
	self.res_fl = self.decoded_fl.replace('decoded','res')
	res =[]
	if enforce_color and (enforce_set is None):
		icols = self.XH[:,-2].astype(int)
		XH = self.XH
		for icol in tqdm(np.unique(icols)):
			inds = np.where(icols==icol)[0]
			Xs = XH[inds,:3]
			Ts = cKDTree(Xs)
			res_ = Ts.query_ball_tree(Ts,dinstance_th)
			res += [inds[r] for r in res_]
	elif enforce_color and (enforce_set is not None):
		ibits = self.XH[:,-1].astype(int)
		isets = ibits//enforce_set
		icols = self.XH[:,-2].astype(int)
		XH = self.XH
		for icol in np.unique(icols):
			for iset in tqdm(np.unique(isets)):
				inds = np.where((icols==icol)&(isets==iset))[0]
				Xs = XH[inds,:3]
				Ts = cKDTree(Xs)
				res_ = Ts.query_ball_tree(Ts,dinstance_th)
				res += [inds[r] for r in res_]
	else:
		XH = self.XH
		Xs = XH[:,:3]
		Ts = cKDTree(Xs)
		res = Ts.query_ball_tree(Ts,dinstance_th)
	print("Calculating lengths of clusters...")
	lens = np.array(list(map(len,res)))
	Mlen = np.max(lens)
	print("Unfolding indexes...")
	res_unfolder = np.concatenate(res)
	print("Saving to file:",self.res_fl)
	self.res_unfolder=res_unfolder
	self.lens=lens
	#np.savez(self.res_fl,res_unfolder=res_unfolder,lens=lens)
	lens =self.lens
	self.res_unfolder = self.res_unfolder[np.repeat(lens, lens)>=nmin_bits]
	self.lens = self.lens[lens>=nmin_bits]
def get_icodesV3(dec,nmin_bits=3,iH=-3):
	import time
	start = time.time()
	lens = dec.lens
	res_unfolder = dec.res_unfolder
	Mlen = np.max(lens)
	print("Calculating indexes within cluster...")
	res_is = np.tile(np.arange(Mlen), len(lens))
	res_is = res_is[res_is < np.repeat(lens, Mlen)]
	print("Calculating index of molecule...")
	ires = np.repeat(np.arange(len(lens)), lens)
	#r0 = np.array([r[0] for r in res for r_ in r])
	print("Calculating index of first molecule...")
	r0i = np.concatenate([[0],np.cumsum(lens)])[:-1]
	r0 = res_unfolder[np.repeat(r0i, lens)]
	print("Total time unfolded molecules:",time.time()-start)
	
	### torch
	ires = torch.from_numpy(ires.astype(np.int64))
	res_unfolder = torch.from_numpy(res_unfolder.astype(np.int64))
	res_is = torch.from_numpy(res_is.astype(np.int64))
	
	import time
	start = time.time()
	print("Computing score...")
	scoreF = torch.from_numpy(dec.XH[:,iH])[res_unfolder]
	print("Total time computing score:",time.time()-start)
	
	
	### organize molecules in blocks for each cluster
	def get_asort_scores():
		val = torch.max(scoreF)+2
		scoreClu = torch.zeros([len(lens),Mlen],dtype=torch.float64)+val
		scoreClu[ires,res_is]=scoreF
		asort = scoreClu.argsort(-1)
		scoreClu = torch.gather(scoreClu,dim=-1,index=asort)
		scoresF2 = scoreClu[scoreClu<val-1]
		return asort,scoresF2
	def get_reorder(x,val=-1):
		if type(x) is not torch.Tensor:
			x = torch.from_numpy(np.array(x))
		xClu = torch.zeros([len(lens),Mlen],dtype=x.dtype)+val
		xClu[ires,res_is] = x
		xClu = torch.gather(xClu,dim=-1,index=asort)
		xf = xClu[xClu>val]
		return xf
	
	
	import time
	start = time.time()
	print("Computing sorting...")
	asort,scoresF2 = get_asort_scores()
	res_unfolder2 = get_reorder(res_unfolder,val=-1)
	del asort
	del scoreF
	print("Total time sorting molecules by score:",time.time()-start)
	
	import time
	start = time.time()
	print("Finding best bits per molecules...")
	
	Rs = dec.XH[:,-1].astype(np.int64)
	Rs = torch.from_numpy(Rs)
	Rs_U = Rs[res_unfolder2]
	nregs,nbits = dec.codes_01.shape
	score_bits = torch.zeros([len(lens),nbits],dtype=scoresF2.dtype)-1
	score_bits[ires,Rs_U]=scoresF2
	
	
	codes_lib = torch.from_numpy(np.array(dec.codes__))
	
	
	codes_lib_01 = torch.zeros([len(codes_lib),nbits],dtype=score_bits.dtype)
	for icd,cd in enumerate(codes_lib):
		codes_lib_01[icd,cd]=1
	codes_lib_01 = codes_lib_01/torch.norm(codes_lib_01,dim=-1)[:,np.newaxis]
	print("Finding best code...")
	batch = 10000
	icodes_best = torch.zeros(len(score_bits),dtype=torch.int64)
	dists_best = torch.zeros(len(score_bits),dtype=torch.float32)
	from tqdm import tqdm
	for i in tqdm(range((len(score_bits)//batch)+1)):
		score_bits_ = score_bits[i*batch:(i+1)*batch]
		if len(score_bits_)>0:
			score_bits__ = score_bits_.clone()
			score_bits__[score_bits__==-1]=0
			score_bits__ = score_bits__/torch.norm(score_bits__,dim=-1)[:,np.newaxis]
			Mul = torch.matmul(score_bits__,codes_lib_01.T)
			max_ = torch.max(Mul,dim=-1)
			icodes_best[i*batch:(i+1)*batch] = max_.indices
			dists_best[i*batch:(i+1)*batch] = 2-2*max_.values
	
	
	keep_all_bits = torch.sum(score_bits.gather(1,codes_lib[icodes_best])>=0,-1)>=nmin_bits
	dists_best_ = dists_best[keep_all_bits]
	score_bits = score_bits[keep_all_bits]
	icodes_best_ = icodes_best[keep_all_bits]
	icodesN=icodes_best_
	
	indexMols_ = torch.zeros([len(lens),nbits],dtype=res_unfolder2.dtype)-1
	indexMols_[ires,Rs_U]=res_unfolder2
	indexMols_ = indexMols_[keep_all_bits]
	indexMols_ = indexMols_.gather(1,codes_lib[icodes_best_])
	
	# make unique
	indexMols_,rinvMols = get_unique_ordered(indexMols_)
	icodesN = icodesN[rinvMols]
	
	XH = torch.from_numpy(dec.XH)
	XH_pruned = XH[indexMols_]
	XH_pruned[indexMols_==-1]=np.nan
	
	dec.dist_best = dists_best_[rinvMols].numpy()
	dec.XH_pruned=XH_pruned.numpy()
	dec.icodesN=icodesN.numpy()
	np.savez_compressed(dec.decoded_fl,XH_pruned=dec.XH_pruned,icodesN=dec.icodesN,gns_names = np.array(dec.gns_names),dist_best=dec.dist_best)
	print("Total time best bits per molecule:",time.time()-start)

# this is where worker script starts
def compute_decoding(save_folder,fov,set_,lib_fl, redo=False):
	dec = decoder_simple(save_folder,fov,set_)
	complete = dec.check_is_complete()
	if complete==0 or redo:
		#compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
		dec = decoder_simple(save_folder,fov=fov,set_=set_)
		dec.get_XH(fov,set_,ncols=3,nbits=100,th_h=3600,tag_keep='_AER_')#number of colors match 
		dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
		dec.load_library(lib_fl,nblanks=-1)
		
		dec.ncols = 3
		get_intersV2(dec,nmin_bits=3,dinstance_th=2,enforce_color=True,enforce_set=None,redo=False)
		get_icodesV3(dec,nmin_bits=3,iH=-3)
		#dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
		#dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=True,redo=True)
		#dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
		#get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=24,is_unique=True)



def get_XH(self,ncols=3,nbits=16,th_h=0,,color_fl=None):
	drifts,all_flds,fov,fl_ref = pickle.load(open(drift_fl,'rb'))
	self.drifts,self.all_flds,self.fov,self.fl_ref = drifts,all_flds,fov,fl_ref

	XH = []
	for iH in tqdm(np.arange(len(all_flds))):
		fld = all_flds[iH]
		for icol in range(ncols):
			Xh = np.load(save_fl,allow_pickle=True)['Xh']
			if len(Xh):
				tzxy = drifts[iH][0]
				ih = get_iH(fld) # get bit
				bit = (ih-1)*ncols+icol
				if len(Xh.shape):
					Xh = Xh[Xh[:,-1]>th_h]
					if len(Xh):

						icolR = np.array([[icol,bit]]*len(Xh))

						if color_fl is not None:
							ms = np.load(color_fl,allow_pickle=True)
							Xh[:,:3] = apply_colorcor(Xh[:,:3],ms[icol])
						Xh[:,:3]+=tzxy# drift correction
						XH_ = np.concatenate([Xh,icolR],axis=-1)
						XH.extend(XH_)
	return np.array(XH)




