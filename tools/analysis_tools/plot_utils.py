# Import dependencies
import cv2
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from mmdet.apis import DetInferencer
import mmcv
import torch

def calculate_metrics_class(result_arr,gt_df, class_pkl, score_thresh, ovthresh,img_name, is_multi_class=True):
	gt = np.array([each[0] for each in gt_df.segmentation])
	if is_multi_class:
		gt_class = [class_pkl[t] for t in gt_df.category_id]
	detected_flag = [0] * len(gt)
	npos_valid = gt.shape[0]
	ttp = 0
	ffp = 0
	ttpc = 0
	ffpc = 0
	for class_num,res in enumerate(result_arr):
		bbox = np.array(result_arr[0]['bboxes'])
		score = np.array(result_arr[0]['scores'])
		idx = np.where(score > score_thresh)[0]
		bboxe = bbox[idx]
		score = score[idx]
		classes = np.asarray(['object'] * len(bbox))
		bbox = bbox.astype('int')
		# init
		nd = 0
		nd += len(idx)
		tp = np.zeros(len(idx))
		fp = np.zeros(len(idx))
		tpc = np.zeros(len(idx))
		fpc = np.zeros(len(idx))
		if len(idx)>0:
			bbox = bbox[idx]
			score = score[idx]
			if is_multi_class:
				class_in = np.array([class_pkl[class_num] for x in range(len(bbox))])
		
			d = 0 # number of prediction bboxes
			BBGT = gt
			BBGT = np.asarray(BBGT, dtype=np.float32)
			bbs = bbox

			if len(BBGT) > 0:
				for iddx, bb in enumerate(bbs):
					bb = np.asarray(bb, dtype=np.float32)

					if BBGT.size > 0:
						# compute overlaps
						# intersection
						ixmin = np.maximum(BBGT[:, 0], bb[0])
						iymin = np.maximum(BBGT[:, 1], bb[1])
						ixmax = np.minimum(BBGT[:, 2], bb[2])
						iymax = np.minimum(BBGT[:, 3], bb[3])
						iw = np.maximum(ixmax - ixmin + 1., 0.)
						ih = np.maximum(iymax - iymin + 1., 0.)
						inters = iw * ih

						# union
						uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
							(BBGT[:, 2] - BBGT[:, 0] + 1.) *
							(BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

						overlaps = inters / uni
						ovmax = np.max(overlaps)
						jmax = np.argmax(overlaps)

					if ovmax > ovthresh:
						if not detected_flag[jmax]:
							if is_multi_class:
								if class_in[iddx]==gt_class[jmax]:
									tpc[d] = 1.
								else:
									fpc[d] = 1.
							tp[d] = 1.
							detected_flag[jmax] = 1
						else:
							fp[d] = 1.
							fpc[d] = 1.
							
					else:
						fp[d] = 1.	
						fpc[d] = 1.			
					d += 1

			else:# if block 36
				num_bbs = len(bbs)
				fp[d: d+num_bbs] = 1	# add false positive flag to each detection
				fpc[d: d+num_bbs] = 1
				d+= num_bbs				# update number of detections	
		else: #if block 16
			ffp += np.sum(fp)
			ttp += np.sum(tp)		

		# for each class
		ffp += np.sum(fp)
		ttp += np.sum(tp)
		ffpc += np.sum(fpc)
		ttpc += np.sum(tpc) 

	rec = ttp / float(npos_valid)
	prec = ttp / np.maximum(ttp + ffp, np.finfo(np.float64).eps)
	if is_multi_class:
		recc = ttpc / float(npos_valid)
		precc = ttpc / np.maximum(ttpc + ffpc, np.finfo(np.float64).eps)
	else:
		recc,precc,ttpc,ffpc = 0,0,0,0
	# return rec, prec, ap
	return {'gpd_det':[img_name,'detection',rec, prec, ttp, ffp, npos_valid],'gpd_class':[img_name,'classification',recc,precc,ttpc,ffpc,npos_valid]}
	

def multi_class_inference(result_arr,score_thresh=0.5,class_pkl={},guarantee_dict={}):
	bboxes = []
	scores = []
	classes = []
	guarantee_list = []
	for class_num,res in enumerate(result_arr):
		bbox = res[:][:, :4]
		score = res[:][:, 4]
		idx = np.where(score > score_thresh)[0]
		if len(idx)>0:
			bbox = bbox[idx]
			score = score[idx]
			class_in = np.array([class_pkl[class_num] for x in range(len(bbox))])
			class_thresh= guarantee_dict[class_in[0]] if class_in[0] in guarantee_dict.keys() else 1
			guarantee = np.array(score>class_thresh, dtype='int')
			bboxes.extend([list(box) for box in bbox])
			scores.extend(list(score))
			classes.extend(list(class_in))
			guarantee_list.extend(list(guarantee))
	torch.cuda.empty_cache()
	return {'boxes' : np.array(bboxes), 'classes' : np.array(classes), 'scores' : np.array(scores), "guarantees": np.array(guarantee_list)} 

def plot_multi_class(multi_dict1, ori_im):
	bboxes, scores, classes, guarantees = multi_dict1['boxes'],multi_dict1['scores'],multi_dict1['classes'],multi_dict1['guarantees']
	
	color = [(255, 0, 0), (0, 0, 225), (0, 200, 225), (255, 0, 255)]
	for box, score, class_value,guarantee in zip(bboxes, scores, classes,guarantees):
		start_point = (int(box[0]), int(box[1]))
		end_point = (int(box[2]), int(box[3]))
		ori_im = cv2.rectangle(ori_im, start_point, end_point, color[1], thickness=4)
		ori_im = cv2.putText(ori_im, '%s:%.2f:%i'%(class_value,score,guarantee), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color[1], 2, cv2.LINE_AA)
	return ori_im

def load_multi_class_var(multi_class_path,guarantee_pkl_path):
	multi_class_dict = pickle.load(open(multi_class_path, 'rb'))
	if type(list(multi_class_dict.keys())[0])!=int:
		multi_class_dict = {v: k for k, v in multi_class_dict.items()}# dict should in this format {0:'class_1',1:'class_2'}
	guarantee_dict = pickle.load(open(guarantee_pkl_path,'rb'))
	return multi_class_dict, guarantee_dict

def make_dir(base_path,save_fol):
	if not os.path.exists(os.path.join(base_path,save_fol)):
		os.makedirs(os.path.join(base_path,save_fol))

def common_load(gt_sheet,annot,no_images):
	df_sheet=pd.read_csv(gt_sheet)
	color = [(255, 0, 0), (0, 0, 225), (0, 200, 225)]
	df = pd.DataFrame(annot['annotations']) # Convert all annotations into dataframe.
	# if None is passed, visualisation of all images will be saved 
	no_images = no_images if no_images and no_images<=len(annot['images']) else len(annot['images'])
	
	return df_sheet,color,df,no_images

def common_load_each_img(annot,i,img_base_path,df_sheet,df):
	img_name = annot['images'][i]['file_name']
	img_path = os.path.join(img_base_path, img_name)
	img_id = annot['images'][i]['id']
	## extracting AOI of that image
	if 'imname' not in df_sheet.columns: 
		img_df = df_sheet[df_sheet['file_name']==img_name]
	else:
		img_df = df_sheet[df_sheet['imname']==f'tagged_raw_images/{img_name}']
	# filter annotations based on img_id
	temp_df = df[df['image_id']==img_id]
	return img_name,img_path, img_df, temp_df

def img_csv_pre_processing(df,is_multi_class):
	d1 = df[df['module']=='detection']
	d1_tp,d1_fp,d1_gt = d1['TP'].sum(),d1['FP'].sum(),d1['GT'].sum() 
	d1_rec,d1_prec = d1_tp/d1_gt,d1_tp/(d1_tp+d1_fp)
	if is_multi_class:
		c1 = df[df['module']=='classification']
		c1_tp,c1_fp,c1_gt = c1['TP'].sum(),c1['FP'].sum(),c1['GT'].sum()
		c1_rec,c1_prec = c1_tp/c1_gt,c1_tp/(c1_tp+c1_fp)
	df.loc[len(df.index)] = ['','detection_total',d1_rec,d1_prec,d1_tp,d1_fp,d1_gt]
	if is_multi_class:
		df.loc[len(df.index)] = ['','classification_total',c1_rec,c1_prec,c1_tp,c1_fp,c1_gt]
	return df

def plot_annots(base_path, img_base_path, annot, gt_sheet, no_images=None, model=None,save_fol="viz_gt",conf_thresh=0.4,label_score_plot=True,gt_box_plot=True,is_multi_class=False,multi_class_path=None,guarantee_pkl_path=None,is_save_csv=True):
	# checks if visualization dir exists or will create.
	make_dir(base_path,save_fol)
	os.chdir(img_base_path)
	# imagewise accuracy reuired var
	img_data = []
	img_columns = ['imname','module','Recall','Precision','TP','FP','GT']
	#Is Multi-Class GPD inference
	if is_multi_class:
		multi_class_dict,guarantee_dict = load_multi_class_var(multi_class_path,guarantee_pkl_path) 
	else:
		multi_class_dict,guarantee_dict = None, None
	# common load
	df_sheet,color,df,no_images = common_load(gt_sheet,annot,no_images)
	for i in tqdm(range(len(annot['images']))[:no_images], ncols=50):
		
		img_name, img_path, img_df, temp_df = common_load_each_img(annot,i,img_base_path,df_sheet,df)
		a_xmin, a_ymin, a_xmax, a_ymax= img_df[['a_xmin', 'a_ymin', 'a_xmax', 'a_ymax']].iloc[0]
		img = mmcv.imread(img_path)
		
		# check if image none
		if img is None:
			print('Could not read image')
		results = model(img)
		result_arr = np.array(results["predictions"])

		if model != None:
			if is_multi_class:
				#call image-level accuracy function
				acc_logger = calculate_metrics_class(result_arr,temp_df, multi_class_dict, conf_thresh, 0.5,img_name,is_multi_class=is_multi_class)
				img_data.append(acc_logger['gpd_det'])
				img_data.append(acc_logger['gpd_class'])
				# multi-class inference
				multi_class_out_dict = multi_class_inference(result_arr,0.5,multi_class_dict,guarantee_dict)
				img = plot_multi_class(multi_class_out_dict,img)
			else:	
				acc_logger = calculate_metrics_class(result_arr,temp_df, multi_class_dict, conf_thresh, 0.5,img_name,is_multi_class=is_multi_class)
				img_data.append(acc_logger['gpd_det'])
				boxes = np.array(result_arr[0]['bboxes'])
				scores = np.array(result_arr[0]['scores'])
				# filter out boxes based on confidence thresh default is 0.4.
				boxes = boxes[np.where(scores>=conf_thresh)]
				scores = scores[np.where(scores>=conf_thresh)]
				for box, score in zip(boxes, scores):
					start_point = (int(box[0]), int(box[1]))
					end_point = (int(box[2]), int(box[3]))
					cv2.rectangle(img, start_point, end_point, color[1], thickness=3)
					if label_score_plot:
						cv2.putText(img, str(round(score,2)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color[1], 2, cv2.LINE_AA)
		if gt_box_plot:
			for index, row in temp_df.iterrows():
				bbox = row['segmentation'][0]
				start_point = (int(bbox[0]),int(bbox[1]))
				end_point = (int(bbox[2]),int(bbox[3]))
				t_point = (int(bbox[0]),int(bbox[3]))
				class_gt = multi_class_dict[row['category_id']] if is_multi_class else ''
				cv2.rectangle(img, start_point, end_point, color[0], thickness=4)
				cv2.putText(img, class_gt, t_point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color[0], 2, cv2.LINE_AA)

		if model != None and label_score_plot:
			cv2.putText(img, 'Predicted', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color[1], 2, cv2.LINE_AA)
		if label_score_plot:
			cv2.putText(img, 'Ground Truth', (30,110), cv2.FONT_HERSHEY_SIMPLEX, 2, color[0], 2, cv2.LINE_AA)
		cv2.rectangle(img, (int(a_xmin),int(a_ymin)), (int(a_xmax),int(a_ymax)), color[2], thickness=2)
		# save image.
		cv2.imwrite(os.path.join(base_path,save_fol,img_name),img)
	# save imagewise accuracy
	if is_save_csv:
		df = pd.DataFrame(img_data,columns=img_columns)
		df = img_csv_pre_processing(df,is_multi_class)
		df.to_csv(os.path.join(base_path,'imagewise_accuracy.csv'),index=False)

if __name__=='__main__':
	# Base path
	__BASE_PATH__ = "/test/maruthi/mmdet_latest/mmdetection1/"
	# GT sheet to extract aoi
	gt_sheet="/test/maruthi/mmdet_latest/mmdetection1/Dataset_Mondelez/112d2733-1432-4562-85ec-4299755b7f19_169738.csv"
	
	# GPD dataset path
	gpd_dataset_path = os.path.join(__BASE_PATH__, 'Dataset_Mondelez/')
	# Images base path
	#img_base_path = os.path.join(__BASE_PATH__, 'tagged_raw_images/')
	img_base_path = "/test/maruthi/mmdet_latest/mmdetection1/Dataset_Mondelez/images/"
	#/home/pdguest/maruthi/mmdet_latest/mmdetection1/Dataset_Mondelez/tagged_raw_images
	## Number of images to visualise: 0 if you want to visualize on all images.
	no_images=10
	# Load test annotations json file
	annot = json.load(open(os.path.join(gpd_dataset_path, 'val_coco.json'), 'rb'))
	model_path = "/test/maruthi/mmdet_latest/mmdetection1/work_dirs/pd_gfl_resnet50/best_coco_bbox_mAP_epoch_16.pth" # None if you want to plot only gt.
	config_gfl = "/test/maruthi/mmdet_latest/mmdetection1/work_dirs/pd_gfl_resnet50/pd_gfl.py"
	#if multi-class GPD True update path else None
	multi_class_path = None 
	guarantee_path = None 
	if model_path==None:
		model = None
		print('Model path not provided. Plotting only GT!')
	else:
		model = DetInferencer(config_gfl, model_path,device='cuda:0')
	# calling main function
	plot_annots(gpd_dataset_path, img_base_path, annot, gt_sheet,no_images, model, conf_thresh=float(input("conf thresh = ")), save_fol=f"viz_gt_",is_multi_class=False,multi_class_path=multi_class_path,guarantee_pkl_path=guarantee_path,is_save_csv=True)