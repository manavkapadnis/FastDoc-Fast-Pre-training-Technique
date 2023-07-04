def custom_loss(similarity, label, sent1_heirarchy_preds, sent2_heirarchy_preds, sent1_heirarchy_ground, sent2_heirarchy_ground,
                      criterion_main, criterion_heirarchy, weight_main, weight1, weight2, weight3, weight4, weight5, weight6, weight7):
          
  loss_main = weight_main * criterion_main(similarity, label)

  #print(sent1_heirarchy_ground.shape[0])
  #print(sent1_heirarchy_ground[:, 0].shape)

  # print(sent1_heirarchy_preds[0])
  # print(sent1_heirarchy_ground[:, 0])

  loss_1_sent1 = weight1* criterion_heirarchy(sent1_heirarchy_preds[0], sent1_heirarchy_ground[:, 0])
  loss_2_sent1 = weight2* criterion_heirarchy(sent1_heirarchy_preds[1], sent1_heirarchy_ground[:, 1])
  loss_3_sent1 = weight3* criterion_heirarchy(sent1_heirarchy_preds[2], sent1_heirarchy_ground[:, 2])
  loss_4_sent1 = weight4* criterion_heirarchy(sent1_heirarchy_preds[3], sent1_heirarchy_ground[:, 3])
  loss_5_sent1 = weight5* criterion_heirarchy(sent1_heirarchy_preds[4], sent1_heirarchy_ground[:, 4])
  loss_6_sent1 = weight6* criterion_heirarchy(sent1_heirarchy_preds[5], sent1_heirarchy_ground[:, 5])
  loss_7_sent1 = weight7* criterion_heirarchy(sent1_heirarchy_preds[6], sent1_heirarchy_ground[:, 6])   

  # print(sent1_heirarchy_preds[4].shape)

  loss_1_sent2 = weight1* criterion_heirarchy(sent2_heirarchy_preds[0], sent2_heirarchy_ground[:, 0])
  loss_2_sent2 = weight2* criterion_heirarchy(sent2_heirarchy_preds[1], sent2_heirarchy_ground[:, 1])
  loss_3_sent2 = weight3* criterion_heirarchy(sent2_heirarchy_preds[2], sent2_heirarchy_ground[:, 2])
  loss_4_sent2 = weight4* criterion_heirarchy(sent2_heirarchy_preds[3], sent2_heirarchy_ground[:, 3])

  #print(sent2_heirarchy_preds[4].shape)

  loss_5_sent2 = weight5* criterion_heirarchy(sent2_heirarchy_preds[4], sent2_heirarchy_ground[:, 4])
  loss_6_sent2 = weight6* criterion_heirarchy(sent2_heirarchy_preds[5], sent2_heirarchy_ground[:, 5])
  loss_7_sent2 = weight7* criterion_heirarchy(sent2_heirarchy_preds[6], sent2_heirarchy_ground[:, 6])
  main_loss = loss_main + loss_1_sent1+ loss_2_sent1+ loss_3_sent1+ loss_4_sent1+ loss_5_sent1+ loss_6_sent1+ loss_7_sent1+ \
               loss_1_sent2+ loss_2_sent2+ loss_3_sent2+ loss_4_sent2+ loss_5_sent2+ loss_6_sent2+ loss_7_sent2 #, requires_grad = True)

  # main_loss = torch.sum(loss_list)
  return main_loss, loss_main, str([loss_1_sent1, loss_2_sent1, loss_3_sent1, loss_4_sent1, loss_5_sent1, loss_6_sent1, loss_7_sent1]), \
  str([loss_1_sent2, loss_2_sent2, loss_3_sent2, loss_4_sent2, loss_5_sent2, loss_6_sent2, loss_7_sent2])

def custom_loss_triplet(anchor_out, pos_out, neg_out, sent1_heirarchy_preds, sent2_heirarchy_preds, sent3_heirarchy_preds, sent1_heirarchy_ground, sent2_heirarchy_ground, sent3_heirarchy_ground,
                      criterion_main, criterion_heirarchy, weight_main, weight1, weight2, weight3, weight4, weight5, weight6, weight7, is_pert_sent = False, criterion_pert = None, anchor_p = None, pos_p = None, neg_p = None, orig_positions_list = None, is_pred_list = None, weight_pert = None):
          
  loss_main = weight_main * criterion_main(anchor_out, pos_out, neg_out)

  #print(sent1_heirarchy_ground.shape[0])
  #print(sent1_heirarchy_ground[:, 0].shape)

  # print(sent1_heirarchy_preds[0])
  # print(sent1_heirarchy_ground[:, 0])

  loss_1_sent1 = weight1* criterion_heirarchy(sent1_heirarchy_preds[0], sent1_heirarchy_ground[0])
  loss_2_sent1 = weight2* criterion_heirarchy(sent1_heirarchy_preds[1], sent1_heirarchy_ground[1])
  loss_3_sent1 = weight3* criterion_heirarchy(sent1_heirarchy_preds[2], sent1_heirarchy_ground[2])
  loss_4_sent1 = weight4* criterion_heirarchy(sent1_heirarchy_preds[3], sent1_heirarchy_ground[3])
  loss_5_sent1 = weight5* criterion_heirarchy(sent1_heirarchy_preds[4], sent1_heirarchy_ground[4])
  loss_6_sent1 = weight6* criterion_heirarchy(sent1_heirarchy_preds[5], sent1_heirarchy_ground[5])
  loss_7_sent1 = weight7* criterion_heirarchy(sent1_heirarchy_preds[6], sent1_heirarchy_ground[6])   

  # print(sent1_heirarchy_preds[4].shape)

  loss_1_sent2 = weight1* criterion_heirarchy(sent2_heirarchy_preds[0], sent2_heirarchy_ground[0])
  loss_2_sent2 = weight2* criterion_heirarchy(sent2_heirarchy_preds[1], sent2_heirarchy_ground[1])
  loss_3_sent2 = weight3* criterion_heirarchy(sent2_heirarchy_preds[2], sent2_heirarchy_ground[2])
  loss_4_sent2 = weight4* criterion_heirarchy(sent2_heirarchy_preds[3], sent2_heirarchy_ground[3])

  #print(sent2_heirarchy_preds[4].shape)

  loss_5_sent2 = weight5* criterion_heirarchy(sent2_heirarchy_preds[4], sent2_heirarchy_ground[4])
  loss_6_sent2 = weight6* criterion_heirarchy(sent2_heirarchy_preds[5], sent2_heirarchy_ground[5])
  loss_7_sent2 = weight7* criterion_heirarchy(sent2_heirarchy_preds[6], sent2_heirarchy_ground[6])

  loss_1_sent3 = weight1* criterion_heirarchy(sent3_heirarchy_preds[0], sent3_heirarchy_ground[0])
  loss_2_sent3 = weight2* criterion_heirarchy(sent3_heirarchy_preds[1], sent3_heirarchy_ground[1])
  loss_3_sent3 = weight3* criterion_heirarchy(sent3_heirarchy_preds[2], sent3_heirarchy_ground[2])
  loss_4_sent3 = weight4* criterion_heirarchy(sent3_heirarchy_preds[3], sent3_heirarchy_ground[3])
  loss_5_sent3 = weight5* criterion_heirarchy(sent3_heirarchy_preds[4], sent3_heirarchy_ground[4])
  loss_6_sent3 = weight6* criterion_heirarchy(sent3_heirarchy_preds[5], sent3_heirarchy_ground[5])
  loss_7_sent3 = weight7* criterion_heirarchy(sent3_heirarchy_preds[6], sent3_heirarchy_ground[6]) 


  main_loss = loss_main + loss_1_sent1+ loss_2_sent1+ loss_3_sent1+ loss_4_sent1+ loss_5_sent1+ loss_6_sent1+ loss_7_sent1+ \
               loss_1_sent2+ loss_2_sent2+ loss_3_sent2+ loss_4_sent2+ loss_5_sent2+ loss_6_sent2+ loss_7_sent2 + \
               loss_1_sent3+ loss_2_sent3+ loss_3_sent3+ loss_4_sent3+ loss_5_sent3+ loss_6_sent3+ loss_7_sent3

  if is_pert_sent:
    pred_list = [anchor_p, pos_p, neg_p]
    # print()
    # print(anchor_p.shape)
    # print()
    for i in range(3):
      for sample_idx in range(is_pred_list[i].shape[0]):
        for idx_pos in range(is_pred_list[i][sample_idx].shape[0]):
          if is_pred_list[i][sample_idx][idx_pos] == 0:
            continue
          main_loss += weight_pert*criterion_pert(pred_list[i][sample_idx][idx_pos], orig_positions_list[i][sample_idx][idx_pos].long())

  # main_loss = torch.sum(loss_list)
  return main_loss#, loss_main, str([loss_1_sent1, loss_2_sent1, loss_3_sent1, loss_4_sent1, loss_5_sent1, loss_6_sent1, loss_7_sent1]), \
  #str([loss_1_sent2, loss_2_sent2, loss_3_sent2, loss_4_sent2, loss_5_sent2, loss_6_sent2, loss_7_sent2])

def custom_loss_quadruplet(anchor_out, near_pos_out, far_pos_out, neg_out, sent1_heirarchy_preds, sent2_heirarchy_preds, sent3_heirarchy_preds, sent4_heirarchy_preds, sent1_heirarchy_ground, sent2_heirarchy_ground, sent3_heirarchy_ground, sent4_heirarchy_ground,
                      criterion_main, criterion_heirarchy, weight_main, weight1, weight2, weight3, weight4, weight5, weight6, weight7):
          
  loss_main = weight_main * (criterion_main(anchor_out, near_pos_out, neg_out) + 0.1*criterion_main(anchor_out, far_pos_out, neg_out))

  #print(sent1_heirarchy_ground.shape[0])
  #print(sent1_heirarchy_ground[:, 0].shape)

  # print(sent1_heirarchy_preds[0])
  # print(sent1_heirarchy_ground[:, 0])

  loss_1_sent1 = weight1* criterion_heirarchy(sent1_heirarchy_preds[0], sent1_heirarchy_ground[:, 0])
  loss_2_sent1 = weight2* criterion_heirarchy(sent1_heirarchy_preds[1], sent1_heirarchy_ground[:, 1])
  loss_3_sent1 = weight3* criterion_heirarchy(sent1_heirarchy_preds[2], sent1_heirarchy_ground[:, 2])
  loss_4_sent1 = weight4* criterion_heirarchy(sent1_heirarchy_preds[3], sent1_heirarchy_ground[:, 3])
  loss_5_sent1 = weight5* criterion_heirarchy(sent1_heirarchy_preds[4], sent1_heirarchy_ground[:, 4])
  loss_6_sent1 = weight6* criterion_heirarchy(sent1_heirarchy_preds[5], sent1_heirarchy_ground[:, 5])
  loss_7_sent1 = weight7* criterion_heirarchy(sent1_heirarchy_preds[6], sent1_heirarchy_ground[:, 6])   

  # print(sent1_heirarchy_preds[4].shape)

  loss_1_sent2 = weight1* criterion_heirarchy(sent2_heirarchy_preds[0], sent2_heirarchy_ground[:, 0])
  loss_2_sent2 = weight2* criterion_heirarchy(sent2_heirarchy_preds[1], sent2_heirarchy_ground[:, 1])
  loss_3_sent2 = weight3* criterion_heirarchy(sent2_heirarchy_preds[2], sent2_heirarchy_ground[:, 2])
  loss_4_sent2 = weight4* criterion_heirarchy(sent2_heirarchy_preds[3], sent2_heirarchy_ground[:, 3])

  #print(sent2_heirarchy_preds[4].shape)

  loss_5_sent2 = weight5* criterion_heirarchy(sent2_heirarchy_preds[4], sent2_heirarchy_ground[:, 4])
  loss_6_sent2 = weight6* criterion_heirarchy(sent2_heirarchy_preds[5], sent2_heirarchy_ground[:, 5])
  loss_7_sent2 = weight7* criterion_heirarchy(sent2_heirarchy_preds[6], sent2_heirarchy_ground[:, 6])

  loss_1_sent3 = weight1* criterion_heirarchy(sent3_heirarchy_preds[0], sent3_heirarchy_ground[:, 0])
  loss_2_sent3 = weight2* criterion_heirarchy(sent3_heirarchy_preds[1], sent3_heirarchy_ground[:, 1])
  loss_3_sent3 = weight3* criterion_heirarchy(sent3_heirarchy_preds[2], sent3_heirarchy_ground[:, 2])
  loss_4_sent3 = weight4* criterion_heirarchy(sent3_heirarchy_preds[3], sent3_heirarchy_ground[:, 3])
  loss_5_sent3 = weight5* criterion_heirarchy(sent3_heirarchy_preds[4], sent3_heirarchy_ground[:, 4])
  loss_6_sent3 = weight6* criterion_heirarchy(sent3_heirarchy_preds[5], sent3_heirarchy_ground[:, 5])
  loss_7_sent3 = weight7* criterion_heirarchy(sent3_heirarchy_preds[6], sent3_heirarchy_ground[:, 6]) 

  loss_1_sent4 = weight1* criterion_heirarchy(sent4_heirarchy_preds[0], sent4_heirarchy_ground[:, 0])
  loss_2_sent4 = weight2* criterion_heirarchy(sent4_heirarchy_preds[1], sent4_heirarchy_ground[:, 1])
  loss_3_sent4 = weight3* criterion_heirarchy(sent4_heirarchy_preds[2], sent4_heirarchy_ground[:, 2])
  loss_4_sent4 = weight4* criterion_heirarchy(sent4_heirarchy_preds[3], sent4_heirarchy_ground[:, 3])
  loss_5_sent4 = weight5* criterion_heirarchy(sent4_heirarchy_preds[4], sent4_heirarchy_ground[:, 4])
  loss_6_sent4 = weight6* criterion_heirarchy(sent4_heirarchy_preds[5], sent4_heirarchy_ground[:, 5])
  loss_7_sent4 = weight7* criterion_heirarchy(sent4_heirarchy_preds[6], sent4_heirarchy_ground[:, 6]) 

  main_loss = loss_main + loss_1_sent1+ loss_2_sent1+ loss_3_sent1+ loss_4_sent1+ loss_5_sent1+ loss_6_sent1+ loss_7_sent1+ \
               loss_1_sent2+ loss_2_sent2+ loss_3_sent2+ loss_4_sent2+ loss_5_sent2+ loss_6_sent2+ loss_7_sent2 + \
               loss_1_sent3+ loss_2_sent3+ loss_3_sent3+ loss_4_sent3+ loss_5_sent3+ loss_6_sent3+ loss_7_sent3 + \
               loss_1_sent4+ loss_2_sent4+ loss_3_sent4+ loss_4_sent4+ loss_5_sent4+ loss_6_sent4+ loss_7_sent4

  # main_loss = torch.sum(loss_list)
  return main_loss#, loss_main, str([loss_1_sent1, loss_2_sent1, loss_3_sent1, loss_4_sent1, loss_5_sent1, loss_6_sent1, loss_7_sent1]), \
  #str([loss_1_sent2, loss_2_sent2, loss_3_sent2, loss_4_sent2, loss_5_sent2, loss_6_sent2, loss_7_sent2])

def custom_loss_only_classfn(sent1_heirarchy_preds, sent2_heirarchy_preds, sent1_heirarchy_ground, sent2_heirarchy_ground, criterion_heirarchy, weight1, weight2, weight3, weight4, weight5, weight6, weight7):
          
  # loss_main = weight_main * criterion_main(similarity, label)

  #print(sent1_heirarchy_ground.shape[0])
  #print(sent1_heirarchy_ground[:, 0].shape)

  # print(sent1_heirarchy_preds[0].shape)

  loss_1_sent1 = weight1* criterion_heirarchy(sent1_heirarchy_preds[0], sent1_heirarchy_ground[:, 0])
  loss_2_sent1 = weight2* criterion_heirarchy(sent1_heirarchy_preds[1], sent1_heirarchy_ground[:, 1])
  loss_3_sent1 = weight3* criterion_heirarchy(sent1_heirarchy_preds[2], sent1_heirarchy_ground[:, 2])
  loss_4_sent1 = weight4* criterion_heirarchy(sent1_heirarchy_preds[3], sent1_heirarchy_ground[:, 3])
  loss_5_sent1 = weight5* criterion_heirarchy(sent1_heirarchy_preds[4], sent1_heirarchy_ground[:, 4])
  loss_6_sent1 = weight6* criterion_heirarchy(sent1_heirarchy_preds[5], sent1_heirarchy_ground[:, 5])
  loss_7_sent1 = weight7* criterion_heirarchy(sent1_heirarchy_preds[6], sent1_heirarchy_ground[:, 6])   

  # print(sent1_heirarchy_preds[4].shape)

  loss_1_sent2 = weight1* criterion_heirarchy(sent2_heirarchy_preds[0], sent2_heirarchy_ground[:, 0])
  loss_2_sent2 = weight2* criterion_heirarchy(sent2_heirarchy_preds[1], sent2_heirarchy_ground[:, 1])
  loss_3_sent2 = weight3* criterion_heirarchy(sent2_heirarchy_preds[2], sent2_heirarchy_ground[:, 2])
  loss_4_sent2 = weight4* criterion_heirarchy(sent2_heirarchy_preds[3], sent2_heirarchy_ground[:, 3])

  #print(sent2_heirarchy_preds[4].shape)

  loss_5_sent2 = weight5* criterion_heirarchy(sent2_heirarchy_preds[4], sent2_heirarchy_ground[:, 4])
  loss_6_sent2 = weight6* criterion_heirarchy(sent2_heirarchy_preds[5], sent2_heirarchy_ground[:, 5])
  loss_7_sent2 = weight7* criterion_heirarchy(sent2_heirarchy_preds[6], sent2_heirarchy_ground[:, 6])
  main_loss = loss_1_sent1+ loss_2_sent1+ loss_3_sent1+ loss_4_sent1+ loss_5_sent1+ loss_6_sent1+ loss_7_sent1+ \
               loss_1_sent2+ loss_2_sent2+ loss_3_sent2+ loss_4_sent2+ loss_5_sent2+ loss_6_sent2+ loss_7_sent2 #, requires_grad = True)

  # main_loss = torch.sum(loss_list)
  return main_loss, str([loss_1_sent1, loss_2_sent1, loss_3_sent1, loss_4_sent1, loss_5_sent1, loss_6_sent1, loss_7_sent1]), \
  str([loss_1_sent2, loss_2_sent2, loss_3_sent2, loss_4_sent2, loss_5_sent2, loss_6_sent2, loss_7_sent2])