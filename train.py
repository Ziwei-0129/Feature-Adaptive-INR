from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import math

from attention import KVMemoryModel
from moe import *


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--dir-weights", required=False, type=str,
                        help="model weights path")
    parser.add_argument("--dir-outputs", required=False, type=str,
                        help="directory for any outputs (ex: images)")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")
    parser.add_argument("--dsp", type=int, default=3,
                        help="dimensions of the simulation parameters (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")
    parser.add_argument("--sp-sr", type=float, default=0.3,
                        help="simulation parameter sampling rate (default: 0.2)")
    parser.add_argument("--sf-sr", type=float, default=0.05,
                        help="scalar field sampling rate (default: 0.02)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--load-batch", type=int, default=1,
                        help="batch size for loading (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="use weighted L1 Loss")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs to train (default: 10000)")
    parser.add_argument("--log-every", type=int, default=1,
                        help="log training status every given number of batches (default: 1)")
    parser.add_argument("--check-every", type=int, default=2,
                        help="save checkpoint every given number of epochs (default: 2)")
    parser.add_argument("--loss", type=str, default='MSE',
                        help="loss function for training (default: MSE)")
    parser.add_argument("--dim1d", type=int, default=32,
                        help="dimension of 1D line for parameter domain")
    parser.add_argument("--num-pairs", type=int, default=2048,      # number of KV spatial feature pairs
                        help="number of KV spatial feature pairs")
    parser.add_argument("--key-dim", type=int, default=3,           # key dim
                        help="dimension of feature for spatial domain in feature grids")
    parser.add_argument("--top-K", type=int, default=16,            # number of top_K keys
                        help="number of top_K keys")
    parser.add_argument("--chunk-size", type=int, default=256,      # size of chunked kv paris
                        help="size of chunked kv paris")
    parser.add_argument("--spatial-fdim", type=int, default=8,      # value dim
                        help="dimension of feature for spatial domain in feature grids")
    parser.add_argument("--param-fdim", type=int, default=8,
                        help="dimension of feature for parameter domain in feature grids")
    parser.add_argument("--dropout", type=int, default=0,
                        help="using dropout layer in MLP, 0: No, other: Yes (default: 0)")
    
    parser.add_argument("--n-experts", type=int, default=2, help="number of experts")
    parser.add_argument("--gpu-id", type=int, default=0, help="id of GPU")
    parser.add_argument("--gpu-ids", type=str, default="0", help="comma separated list of GPU ids to use")
    parser.add_argument("--mlp-encoder-dim", type=int, default=64, help="dimension of feat structure")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--lr-mlp", type=float, default=1e-4, help="encoder MLP learning rate (default: 1e-4)")
    parser.add_argument("--lr-gate", type=float, default=1e-4, help="gate MLP learning rate (default: 1e-4)")
    
    parser.add_argument("--alpha", type=float, default=0.0, help="load balance loss weighting")
    parser.add_argument("--gate-res", type=int, default=16, help="resolution of gate")
    return parser.parse_args()

def main(args):
    # log hyperparameters
    # print(args)
    num_feats = args.num_pairs
    key_dim = args.key_dim
    top_K = args.top_K
    chunk_size = args.chunk_size
    
    n_experts = args.n_experts
    mlp_encoder_dim = args.mlp_encoder_dim
    num_hidden_layers = args.num_hidden_layers
    gate_res = args.gate_res

    network_str = f'mpaso_kv{num_feats}_MLP_dim{mlp_encoder_dim}_{num_hidden_layers}hLayers_keyDim{key_dim}_valDim{args.spatial_fdim}_{args.dim1d}line_{args.param_fdim}pDim_top{top_K}_M{n_experts}_alpha{args.alpha}_gateRes{gate_res}_seed{args.seed}'
    lr_mlp = args.lr_mlp
    lr_gate = args.lr_gate
   
    args.dir_weights = os.path.join("model_weights", network_str)
    args.dir_outputs = os.path.join("outputs", network_str)
    if not os.path.exists(args.dir_weights):
        os.makedirs(args.dir_weights, exist_ok=True)
        os.makedirs(args.dir_outputs, exist_ok=True)
 

    nEnsemble = 4
    data_size = 11845146
    num_sf_batches = math.ceil(nEnsemble * data_size * args.sf_sr / args.batch_size)
    num_sp_sampling = math.ceil(70 * args.sp_sr)
          
    if args.dropout != 0:
        network_str += '_dp'

    # Device setup
    # device = pytorch_device_config(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    device_ids = list(map(int, args.gpu_ids.split(',')))
    print(f"Using GPUs: {device_ids}")
    
    
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fh = open(os.path.join(args.root, "train", "names.txt"))
    filenames = []
    for line in fh:
        filenames.append(line.replace("\n", ""))

    params_arr = np.load(os.path.join(args.root, "train/params.npy"))
    coords = np.load(os.path.join(args.root, "sphereCoord.npy"))
    coords = coords.astype(np.float32)
    data_dicts = []
    for idx in range(len(filenames)):
        # params min [0.0, 300.0, 0.25, 100.0, 1]
        #        max [5.0, 1500.0, 1.0, 300.0, 384]
        params = np.array(params_arr[idx][1:])
        params = (params.astype(np.float32) - np.array([0.0, 300.0, 0.25, 100.0], dtype=np.float32)) / \
                 np.array([5.0, 1200.0, .75, 200.0], dtype=np.float32)
        d = {'file_src': os.path.join(args.root, "train", filenames[idx]), 'params': params}
        data_dicts.append(d)

    lat_min, lat_max = -np.pi / 2, np.pi / 2
    coords[:,0] = (coords[:,0] - (lat_min + lat_max) / 2.0) / ((lat_max - lat_min) / 2.0)
    lon_min, lon_max = 0.0, np.pi * 2
    coords[:,1] = (coords[:,1] - (lon_min + lon_max) / 2.0) / ((lon_max - lon_min) / 2.0)
    depth_min, depth_max = 0.0, np.max(coords[:,2])
    coords[:,2] = (coords[:,2] - (depth_min + depth_max) / 2.0) / ((depth_max - depth_min) / 2.0)

    #########################################################################################################
    
    " Manager network "
    manager_net = Manager(resolution=gate_res, n_experts=n_experts, device=device)
    
    feat_shapes = np.ones(4, dtype=np.int32) * args.dim1d
    inr_fg = KVMemoryModel(feat_shapes, num_entries=num_feats, key_dim=key_dim, feature_dim_3d=args.spatial_fdim, 
                feature_dim_1d=args.param_fdim, top_K=top_K, chunk_size=chunk_size, 
                num_hidden_layers=num_hidden_layers, mlp_encoder_dim=mlp_encoder_dim,
                n_experts=n_experts, manager_net=manager_net)
    
    # Total Trainable Parameters Only:
    trainable_params = sum(p.numel() for p in inr_fg.parameters() if p.requires_grad)
    print(f"\nTrainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n\n")

    # Use DataParallel for multi-GPU
    inr_fg = torch.nn.DataParallel(inr_fg, device_ids=device_ids)
    inr_fg.to(device)
    
    if args.start_epoch > 0:
        inr_fg.module.load_state_dict(torch.load(os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(args.start_epoch) + ".pth")))
    
    # optimizer = torch.optim.Adam(inr_fg.parameters(), lr=args.lr)    
    encoder_mlp_params = set(inr_fg.module.encoder_mlp_list.parameters())
    gating_mlp_params = set(inr_fg.module.manager_net.parameters())
    other_parameters = (param for param in inr_fg.module.parameters() if param not in encoder_mlp_params and \
                        param not in gating_mlp_params)
    optimizer = torch.optim.Adam([
        {'params': inr_fg.module.manager_net.parameters(), 'lr': lr_gate},
        {'params': inr_fg.module.encoder_mlp_list.parameters(), 'lr': lr_mlp},
        {'params': other_parameters, 'lr': args.lr},
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_train_loss = float('inf')
    early_stop_patience = 10
    patience_counter = 0
    
    
    if args.loss == 'MSE':
        print('Use MSE Loss')
        criterion = torch.nn.MSELoss()
    elif args.loss == 'L1':
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()
    else:
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()

    #####################################################################################

    losses = []

    dmin = -1.93
    dmax = 30.36
    num_bins = 10
    bin_width = 1.0 / num_bins
    max_binidx_f = float(num_bins-1)
    batch_size_per_field = args.batch_size // nEnsemble
    nEnsembleGroups_per_epoch = (len(data_dicts)+nEnsemble-1) // nEnsemble
    coords_torch = torch.from_numpy(coords)

    sfimps_np = np.load(os.path.join('outputs', 'mpaso_ensemble_member_importances.npy'))
    sfimps = torch.from_numpy(sfimps_np)

    #####################################################################################

    def imp_func(data, minval, maxval, bw, maxidx):
        freq = None
        nBlocks = 2
        block_size = data_size // nBlocks
        for bidx in range(nBlocks):
            block_freq = torch.histc(data[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=minval, max=maxval).type(torch.long)
            if freq is None:
                freq = block_freq
            else:
                freq += block_freq
        freq = freq.type(torch.double)
        importance = 1. / freq
        importance_idx = torch.clamp((data - minval) / bw, min=0.0, max=maxidx).type(torch.long)
        return importance, importance_idx

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print('epoch {0}'.format(epoch+1))
        tstart1 = time.time()
        total_loss = 0
        e_rndidx = torch.multinomial(sfimps, nEnsembleGroups_per_epoch * nEnsemble, replacement=True)
        for egidx in range(nEnsembleGroups_per_epoch):
            tstart = time.time()
            scalar_fields = []
            sample_weights_arr = []
            params_batch = None
            errsum = 0
            # Load and compute importance map
            for eidx in range(nEnsemble):
                curr_scalar_field = ReadMPASOScalar(data_dicts[e_rndidx[egidx*nEnsemble + eidx]]['file_src'])
                curr_scalar_field = (curr_scalar_field-dmin) / (dmax-dmin)
                curr_scalar_field = torch.from_numpy(curr_scalar_field)
                curr_params = data_dicts[e_rndidx[egidx*nEnsemble + eidx]]['params'].reshape(1,4)
                curr_params = torch.from_numpy(curr_params)
                curr_params_batch = curr_params.repeat(batch_size_per_field, 1)
                if params_batch is None:
                    params_batch = curr_params_batch
                else:
                    params_batch = torch.cat((params_batch, curr_params_batch), 0)
                curr_imp, curr_impidx = imp_func(curr_scalar_field, 0.0, 1.0, bin_width, max_binidx_f)
                curr_sample_weights = curr_imp[curr_impidx]
                
                scalar_fields.append(curr_scalar_field)
                sample_weights_arr.append(curr_sample_weights)
            params_batch = params_batch.to(device)
            # Train
            for field_idx in range(num_sf_batches):
                coord_batch = None
                value_batch = None
                for eidx in range(nEnsemble):
                    #####
                    rnd_idx = torch.multinomial(sample_weights_arr[eidx], batch_size_per_field, replacement=True)
                    ######
                    if coord_batch is None:
                        coord_batch, value_batch = coords_torch[rnd_idx], scalar_fields[eidx][rnd_idx]
                    else:
                        coord_batch, value_batch = torch.cat((coord_batch, coords_torch[rnd_idx]), 0), torch.cat((value_batch, scalar_fields[eidx][rnd_idx]), 0)
                # model outputs are float32 but mpaso values are float64
                value_batch = value_batch.reshape(len(value_batch), 1).type(torch.float32)
                coord_batch = coord_batch.to(device)
                value_batch = value_batch.to(device)
                
                model_output, probs = inr_fg(torch.cat((coord_batch, params_batch), 1), tau=1.0)
                loss = criterion(model_output, value_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_mean_loss = loss.data.cpu().numpy()
                errsum += batch_mean_loss * nEnsemble * batch_size_per_field
                total_loss += batch_mean_loss
            tend = time.time()
            mse = errsum / (nEnsemble * batch_size_per_field * num_sf_batches)
            curr_psnr = - 10. * np.log10(mse)
            print('Training time: {0:.4f} for {1} data points x {2} batches, approx PSNR = {3:.4f}'\
                  .format(tend-tstart, nEnsemble * batch_size_per_field, num_sf_batches, curr_psnr))
        losses.append(total_loss)    
        
        # ---------- Early Stopping ----------
        avg_loss = total_loss / (nEnsembleGroups_per_epoch)
        
        # ----- Learning Rate Decay -----
        scheduler.step(avg_loss)
        
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            patience_counter = 0
            torch.save(inr_fg.module.state_dict(), os.path.join(args.dir_weights, "best_model.pth"))
            print("=> saving BEST model at epoch {}".format(epoch + 1))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered after {} epochs with no improvement.".format(early_stop_patience))
                break
            

        if (epoch+1) % args.log_every == 0:
            print('epoch {0}, loss = {1}'.format(epoch+1, total_loss))
            print("====> Epoch: {0} Average {1} loss: {2:.4f}".format(epoch+1, args.loss, total_loss / num_sp_sampling))
            plt.plot(losses)

            plt.savefig(args.dir_outputs + '/mpaso_fg_inr_loss_' + network_str + '.jpg')
            plt.clf()

        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(inr_fg.module.state_dict(),
                       os.path.join(args.dir_weights, "fg_model_" + network_str + '_'+ str(epoch+1) + ".pth"))

if __name__ == '__main__':
    main(parse_args())