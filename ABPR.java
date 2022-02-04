import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.Map.Entry;

public class ABPR
{
    // =============================================================
    // === Configurations
    // the number of latent dimensions
    public static int d = 20;

    // tradeoff $\alpha_u$
    public static float alpha_u = 0.01f;
    // tradeoff $\alpha_v$
    public static float alpha_v = 0.01f;
    // tradeoff $\beta_v$
    public static float beta_v = 0.01f;
    // tradeoff $\beta_u$
    public static float beta_u = 0.01f;

    // learning rate $\gamma$
    public static float gamma = 0.01f;

    // === the default sizes of user-group
    public static int groupsizeP = 3; // interacted user-group P
    public static int groupsizeN = 3; // un-interacted user-group N

    // === input data files
    public static String fnTrainData = "";
    public static String fnTestData = "";

    //
    public static int n = 0; // number of users
    public static int m = 0; // number of items
    public static int num_train = 0; // number of the total (user, item) pairs in training data

    //
    public static int num_iterations = 1000; // scan number over the whole data
    public static int num_iterations_perTest = -1; // test the performance every num_iterations_perTest iterations

    // === loss
    public static String loss = "ABPRMGO"; // default loss function
    public static boolean flagABPRMGO = false;

    // === evaluation
    //
    public static int topK = 5; // top k in evaluation
    // =============================================================


    // =============================================================
    // === training data
    public static HashMap<Integer, HashSet<Integer>> TrainData = new HashMap<Integer, HashSet<Integer>>(); // user-item pairs
    public static HashMap<Integer, HashSet<Integer>> TrainData_item2user = new HashMap<Integer, HashSet<Integer>>(); // item-user pairs

    // === training data used for uniformly random sampling
    public static int[] indexUserTrain; // start from index "0", used to uniformly sample (u, i) pair
    public static int[] indexItemTrain; // start from index "0", used to uniformly sample (u, i) pair

    // === test data
    public static HashMap<Integer, HashSet<Integer>> TestData = new HashMap<Integer, HashSet<Integer>>();

    // === the whole set of items
    public static HashSet<Integer> itemSetWhole = new HashSet<Integer>();

    // === some statistics, start from index "1"
    public static int[] userRatingNumTrain; // to add up the related records of each user
    public static int[] itemRatingNumTrain; // to add up the related records of each item

    // === model parameters to learn, start from index "1"
    public static float[][] U; // latent feature vector of user
    public static float[][] V; // latent feature vector of item
    public static float[] biasV;  // bias of item
    public static float[] biasU;  // bias of user
    // =============================================================


    // =============================================================
    public static void main(String[] args) throws Exception
    {
        // =========================================================
        // === Read the configurations
        for (int k=0; k < args.length; k++)
        {
            if (args[k].equals("-d")) d = Integer.parseInt(args[++k]);
            else if (args[k].equals("-alpha_u")) alpha_u = Float.parseFloat(args[++k]);
            else if (args[k].equals("-alpha_v")) alpha_v = Float.parseFloat(args[++k]);
            else if (args[k].equals("-beta_u")) beta_u = Float.parseFloat(args[++k]);
            else if (args[k].equals("-beta_v")) beta_v = Float.parseFloat(args[++k]);
            else if (args[k].equals("-gamma")) gamma = Float.parseFloat(args[++k]);
            else if (args[k].equals("-fnTrainData")) fnTrainData = args[++k];
            else if (args[k].equals("-fnTestData")) fnTestData = args[++k];
            else if (args[k].equals("-n")) n = Integer.parseInt(args[++k]);
            else if (args[k].equals("-m")) m = Integer.parseInt(args[++k]);
            else if (args[k].equals("-groupsizeP")) groupsizeP = Integer.parseInt(args[++k]);
            else if (args[k].equals("-groupsizeN")) groupsizeN = Integer.parseInt(args[++k]);
            else if (args[k].equals("-num_iterations")) num_iterations = Integer.parseInt(args[++k]);
            else if (args[k].equals("-num_iterations_perTest")) num_iterations_perTest = Integer.parseInt(args[++k]);
            else if (args[k].equals("-loss")) loss = args[++k];
            else if (args[k].equals("-topK")) topK = Integer.parseInt(args[++k]);
        }

        if(loss.toLowerCase().equals("ABPRMGO".toLowerCase()))   // group vs one
        {
            flagABPRMGO = true;
        }

        // === Print the configurations
        System.out.println(Arrays.toString(args));

        System.out.println("d: " + Integer.toString(d));
        System.out.println("alpha_u: " + Float.toString(alpha_u));
        System.out.println("alpha_v: " + Float.toString(alpha_v));
        System.out.println("beta_u: " + Float.toString(beta_u));
        System.out.println("beta_v: " + Float.toString(beta_v));
        System.out.println("gamma: " + Float.toString(gamma));

        System.out.println("fnTrainData: " + fnTrainData);
        System.out.println("fnTestData: " + fnTestData);
        System.out.println("n: " + Integer.toString(n));
        System.out.println("m: " + Integer.toString(m));

        System.out.println("groupsizeP: " + Integer.toString(groupsizeP));
        System.out.println("groupsizeN: " + Integer.toString(groupsizeN));
        System.out.println("loss: " + loss);
        System.out.println("flagABPRMGO: "+ flagABPRMGO);
        System.out.println("num_iterations: " + Integer.toString(num_iterations));
        System.out.println("num_iterations_perTest: " + Integer.toString(num_iterations_perTest));

        System.out.println("topK: " + Integer.toString(topK));
        // =========================================================


        // =========================================================
        // === some statistics
        itemRatingNumTrain = new int[m+1];
        userRatingNumTrain = new int[n+1];

        // =========================================================
        // === Locate memory for the data structure of the model parameters
        U = new float[n+1][d];
        V = new float[m+1][d];
        biasV = new float[m+1];
        biasU = new float[n+1];

        // =========================================================
        // === Step 1: Read data
        readData();

        // =========================================================
        // === Step 2: Initialization of U, V, bias
        initialize();

        // =========================================================
        // === Step 3: Training
        train();

        // =========================================================
        // === Step 4: Prediction and Evaluation
        testRanking(TestData);
    }


    // =============================================================
    public static void readData() throws Exception
    {
        // =========================================================
        BufferedReader br = new BufferedReader(new FileReader(fnTrainData));
        String line = null;
        while ((line = br.readLine())!=null)
        {
            String[] terms = line.split("\\s+|,|;");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);

            // === add to the whole item set
            itemSetWhole.add(itemID);

            // ===
            userRatingNumTrain[userID] += 1; // to add up the related records of each user
            itemRatingNumTrain[itemID] += 1; // to add up the related records of each item

            // ===
            num_train += 1; // the number of total user-item pairs

            // === TrainData: user->items
            if(TrainData.containsKey(userID))
            {
                HashSet<Integer> itemSet = TrainData.get(userID);
                itemSet.add(itemID);
                TrainData.put(userID, itemSet);
            }
            else
            {
                HashSet<Integer> itemSet = new HashSet<Integer>();
                itemSet.add(itemID);
                TrainData.put(userID, itemSet);
            }

            // === TrainData: item->users
            if(TrainData_item2user.containsKey(itemID))
            {
                HashSet<Integer> userGroup = TrainData_item2user.get(itemID);
                userGroup.add(userID);
                TrainData_item2user.put(itemID, userGroup);
            }
            else
            {
                HashSet<Integer> userGroup = new HashSet<Integer>();
                userGroup.add(userID);
                TrainData_item2user.put(itemID, userGroup);
            }
        }
        br.close();
        // =========================================================

        // =========================================================
        br = new BufferedReader(new FileReader(fnTestData));
        line = null;
        while ((line = br.readLine())!=null)
        {
            String[] terms = line.split("\\s+|,|;");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);

            // === add to the whole item set
            itemSetWhole.add(itemID);

            // === test data
            if(TestData.containsKey(userID))
            {
                HashSet<Integer> itemSet = TestData.get(userID);
                itemSet.add(itemID);
                TestData.put(userID, itemSet);
            }
            else
            {
                HashSet<Integer> itemSet = new HashSet<Integer>();
                itemSet.add(itemID);
                TestData.put(userID, itemSet);
            }
        }
        br.close();
        // =========================================================
    }


    // =============================================================
    public static void initialize()
    {
        // =========================================================
        // === initialization of U and V
        for (int u=1; u<n+1; u++)
        {
            for (int f=0; f<d; f++)
            {
                U[u][f] = (float) ( (Math.random()-0.5)*0.01 );
            }
        }

        // ===
        for (int i=1; i<m+1; i++)
        {
            for (int f=0; f<d; f++)
            {
                V[i][f] = (float) ( (Math.random()-0.5)*0.01 );
            }
        }
        // =========================================================

        // =========================================================
        // === calculation of the global average rating
        float g_avg = 0;
        for (int i=1; i<m+1; i++)
        {
            g_avg += itemRatingNumTrain[i];
        }
        g_avg = g_avg/n/m;

        // === biasV[i] represents the popularity of the item i, which is initialized to [0,1]
        for (int i=1; i<m+1; i++)
        {
            biasV[i]= (float) itemRatingNumTrain[i] / n - g_avg;
        }

        // === biasU[n] represents the popularity of the user u, which is initialized to [0,1]
        for (int u=1; u<n+1; u++)
        {
            biasU[u]= (float) userRatingNumTrain[u] / m - g_avg;
        }
        // =========================================================
    }


    // =============================================================
    public static void train() throws FileNotFoundException
    {
        // === construct indexUserTrain and indexItemTrain
        indexUserTrain = new int[num_train];
        indexItemTrain = new int[num_train];

        int idx = 0;
        for(int u=1; u<=n; u++)
        {
            // === check whether the "user $u$" is in the training Data
            if (!TrainData.containsKey(u))
            {
                continue;
            }

            // === get a copy of the Data in indexUserTrain and indexItemTrain
            HashSet<Integer> itemSet = new HashSet<Integer>();
            if (TrainData.containsKey(u))
            {
                itemSet = TrainData.get(u);
            }

            for(int i : itemSet)
            {
                indexUserTrain[idx] = u;
                indexItemTrain[idx] = i;
                idx += 1;
            }
        }

        for (int iter = 0; iter < num_iterations; iter++)
        {
            // ---
            if( num_iterations_perTest>0 && iter>0 && iter%num_iterations_perTest==0 )
            {
                System.out.print( "Iter: " + Integer.toString(iter) + " | ");
                testNDCG(TestData);
            }

            for (int iter_rand = 0; iter_rand < num_train; iter_rand++)
            {

                // === sample a user-item pair ("user $u$","item $i$")
                idx = (int) Math.floor(Math.random() * num_train);
                int u = indexUserTrain[idx];
                int i = indexItemTrain[idx];

                // ---
                HashSet<Integer> itemSet_u = TrainData.get(u); // the related item-set of "user $u$"
                int j = i;
                while(true)
                {
                    // --- randomly sample an "item $j$", Math.random(): [0.0, 1.0)
                    j = (int) Math.floor(Math.random() * m) + 1;

                    if (itemSetWhole.contains(j) && !itemSet_u.contains(j) )
                    {
                        break;
                    }
                    else
                    {
                        continue;
                    }
                }

                // ---
                if (flagABPRMGO)
                {
                    ABPRMGO(u,i,j);
                }
            }
        }
    }
    // =============================================================

    // =============================================================
    public static void ABPRMGO(int u, int i, int j) { // group vs one
        // =========================================================
        HashSet<Integer> userGroup_i = TrainData_item2user.get(i); // the related user group of "item $i$"
        int userGroupSize = userGroup_i.size();
        List<Integer> list = new ArrayList<Integer>(userGroup_i);

        // =========================================================
        // === randomly sample groupsizeN users
        HashSet<Integer> userGroupN = new HashSet<Integer>();
        int k = 0;
        while (k < groupsizeN) {
            // === randomly sample a user $w$, Math.random(): [0.0, 1.0)
            int w = (int) Math.floor(Math.random() * n) + 1;
            if (TrainData.containsKey(w) && !userGroup_i.contains(w) && !userGroupN.contains(w)) {
                userGroupN.add(w);
                k += 1;
            } else {
                continue;
            }
        }
        int userGroupSizeN = userGroupN.size();

        // =========================================================
        // === randomly sample groupsizeP users
        HashSet<Integer> userGroupP = new HashSet<Integer>();
        userGroupP.add(u); // add "user $u$" to user-group P
        k = 1;
        while (k < groupsizeP && k < userGroupSize) {
            // === randomly sample a user $u'(denote as u1)$, Math.random(): [0.0, 1.0)
            int t = (int) Math.floor(Math.random() * userGroupSize);
            int u1 = list.get(t);

            if (!userGroupP.contains(u1)) {
                userGroupP.add(u1);
                k += 1;
            } else {
                continue;
            }
        }
        int userGroupSizeP = userGroupP.size();

        // =========================================================
        // === calculation of the overall preference of "user-group $P$"
        float r_iP = 0f;
        float[] U_P = new float[d];
        for (int u1 : userGroupP)
        {
            for (int f=0; f<d; f++)
            {
                U_P[f] += U[u1][f]/userGroupSizeP;
            }
            r_iP += (biasV[i]+biasU[u1])/userGroupSizeP;
        }
        for (int f=0; f<d; f++)
        {
            r_iP += V[i][f]*U_P[f];
        }

        // === gradient of the vertical preference differences
        HashMap<Integer, Float> user2Prediction = new HashMap<Integer, Float>();
        for (int w : userGroupN) {
            float r_iw = biasV[i] + biasU[w];
            for (int f = 0; f < d; f++) {
                r_iw += U[w][f] * V[i][f];
            }
            float r_iPw = r_iP - r_iw;
            float EXP_r_iPw = (float) Math.pow(Math.E, r_iPw);
            float loss_iPw = - 1f / (1f + EXP_r_iPw) / userGroupSizeN;
            user2Prediction.put(w, loss_iPw);
        }

        // === gradient of the horizontal preference difference
        float r_uij = 0f;
        r_uij = biasV[i] - biasV[j];
        for (int f = 0; f < d; f++) {
            r_uij += U[u][f] * (V[i][f] - V[j][f]);
        }
        float EXP_r_uij = (float) Math.pow(Math.E, r_uij);
        float loss_uij = -1f / (1f + EXP_r_uij);

        // =========================================================
        // === update $U_{u\cdot}$
        for (int u1 : userGroupP) {
            float[] grad_Uu = new float[d];
            for(int w : userGroupN){
                float loss_iPw = user2Prediction.get(w);
                for (int f = 0; f < d; f++) {
                    grad_Uu[f] += loss_iPw * (V[i][f]/userGroupSizeP);
                }
            }
            if (u1 == u) {
                for (int f = 0; f < d; f++) {
                    grad_Uu[f] += loss_uij * (V[i][f] - V[j][f]);
                }
            }

            for (int f = 0; f < d; f++) {
                U[u1][f] = U[u1][f] - gamma * (grad_Uu[f] + alpha_u * U[u1][f]);
            }
        }

        // =========================================================
        // === update $U_{w\cdot}$
        for (int w : userGroupN) {
            float loss_iPw = user2Prediction.get(w);
            for (int f = 0; f < d; f++) {
                U[w][f] = U[w][f] - gamma * (loss_iPw * (-V[i][f]) + alpha_u * U[w][f]);
            }
        }

        // =========================================================
        // === update $V_{i\cdot}$
        float[] grad_Vi = new float[d];
        for (int w : userGroupN) {
            float loss_iPw = user2Prediction.get(w);
            for (int f = 0; f < d; f++) {
                grad_Vi[f] += loss_iPw * (U_P[f] - U[w][f]);
            }
        }

        for (int f = 0; f < d; f++) {
            grad_Vi[f] += loss_uij * U[u][f];
        }

        for (int f = 0; f < d; f++) {
            V[i][f] = V[i][f] - gamma * (grad_Vi[f] + alpha_v * V[i][f]);
        }

        // =========================================================
        // === update $V_{j\cdot}$
        for (int f = 0; f < d; f++) {
            V[j][f] = V[j][f] - gamma * (loss_uij * (-U[u][f]) + alpha_v * V[j][f]);
        }

        // =========================================================
        // === update $b_i$
        biasV[i] = biasV[i] - gamma * (loss_uij + beta_v * biasV[i]);

        // === update $b_j$
        biasV[j] = biasV[j] - gamma * (loss_uij * (-1) + beta_v * biasV[j]);

        // === update $b_u$
        for (int u1 : userGroupP) {
            float grad_bu = 0f;
            for (int w : userGroupN) {
                float loss_iPw = user2Prediction.get(w);
                grad_bu += loss_iPw / userGroupSizeP;
            }
            biasU[u1] = biasU[u1] - gamma * (grad_bu + beta_u * biasU[u1]);
        }

        // === update $b_w$
        for (int w : userGroupN) {
            float loss_iPw = user2Prediction.get(w);
            biasU[w] = biasU[w] - gamma * (loss_iPw * (-1) + beta_u * biasU[w]);
        }
    }

    // =============================================================
    @SuppressWarnings("unchecked")
    public static void testRanking(HashMap<Integer, HashSet<Integer>> TestData)
    {
        // TestData: user->items
        // =========================================================
        float[] PrecisionSum = new float[topK+1];
        float[] RecallSum = new float[topK+1];
        float[] F1Sum = new float[topK+1];
        float[] NDCGSum = new float[topK+1];
        float[] OneCallSum = new float[topK+1];

        // === calculate the best DCG, which can be used later
        float[] DCGbest = new float[topK+1];
        for (int k=1; k<=topK; k++)
        {
            DCGbest[k] = DCGbest[k-1];
            DCGbest[k] += 1/Math.log(k+1);
        }

        // === number of test cases
        int UserNum_TestData = TestData.keySet().size();

        for(int u=1; u<=n; u++)
        {
            // === check whether the user $u$ is in the test set
            if (!TestData.containsKey(u))
            {
                continue;
            }

            // ===
            HashSet<Integer> itemSet_u_TrainData = new HashSet<Integer>();
            if (TrainData.containsKey(u))
            {
                itemSet_u_TrainData = TrainData.get(u);
            }
            HashSet<Integer> itemSet_u_TestData = TestData.get(u);

            // === the number of preferred items of user $u$ in the test data
            int ItemNum_u_TestData = itemSet_u_TestData.size();

            // =========================================================
            // === prediction
            HashMap<Integer, Float> item2Prediction = new HashMap<Integer, Float>();
            item2Prediction.clear();

            for(int i=1; i<=m; i++)
            {
                if ( !itemSetWhole.contains(i) || itemSet_u_TrainData.contains(i) )
                    continue;

                // === prediction via inner product
                float pred = biasV[i] +biasU[u];
                for (int f=0; f<d; f++)
                {
                    pred += U[u][f]*V[i][f];
                }
                item2Prediction.put(i, pred);
            }
            // === sort
            List<Entry<Integer,Float>> listY =
                    new ArrayList<Entry<Integer,Float>>(item2Prediction.entrySet());
            Collections.sort(listY, new Comparator<Entry<Integer,Float>>()
            {
                public int compare( Entry<Integer, Float> o1, Entry<Integer, Float> o2 )
                {
                    return o2.getValue().compareTo( o1.getValue() );
                }
            });

            // =========================================================
            // === Extract the topK recommended items
            int k=1;
            int[] TopKResult = new int [topK+1];
            Iterator<Entry<Integer, Float>> iter = listY.iterator();
            while (iter.hasNext())
            {
                if(k>topK)
                    break;

                Entry<Integer, Float> entry = (Entry<Integer, Float>) iter.next();
                int itemID = entry.getKey();
                TopKResult[k] = itemID;
                k++;
            }
            // === TopK evaluation
            int HitSum = 0;
            float[] DCG = new float[topK+1];
            float[] DCGbest2 = new float[topK+1];
            for(k=1; k<=topK; k++)
            {
                // ===
                DCG[k] = DCG[k-1];
                int itemID = TopKResult[k];
                if ( itemSet_u_TestData.contains(itemID) )
                {
                    HitSum += 1;
                    DCG[k] += 1 / Math.log(k+1);
                }
                // === precision, recall, F1, 1-call
                float prec = (float) HitSum / k;
                float rec = (float) HitSum / ItemNum_u_TestData;
                float F1 = 0;
                if (prec+rec>0)
                {
                    F1 = 2 * prec*rec / (prec+rec);
                }
                PrecisionSum[k] += prec;
                RecallSum[k] += rec;
                F1Sum[k] += F1;
                // === in case the the number relevant items is smaller than k
                if (itemSet_u_TestData.size()>=k)
                {
                    DCGbest2[k] = DCGbest[k];
                }
                else
                {
                    DCGbest2[k] = DCGbest2[k-1];
                }
                NDCGSum[k] += DCG[k]/DCGbest2[k];
                // ===
                OneCallSum[k] += HitSum>0 ? 1:0;
            }
        }

        // =========================================================
        // === the number of users in the test data
        System.out.println( "The number of users in the test data: " + Integer.toString(UserNum_TestData) );

        // === precision@k
        for(int k=1; k<=topK; k++)
        {
            float prec = PrecisionSum[k]/UserNum_TestData;
            System.out.println("Prec@"+Integer.toString(k)+":"+Float.toString(prec));
        }
        // === recall@k
        for(int k=1; k<=topK; k++)
        {
            float rec = RecallSum[k]/UserNum_TestData;
            System.out.println("Rec@"+Integer.toString(k)+":"+Float.toString(rec));
        }
        // === F1@k
        for(int k=1; k<=topK; k++)
        {
            float F1 = F1Sum[k]/UserNum_TestData;
            System.out.println("F1@"+Integer.toString(k)+":"+Float.toString(F1));
        }
        // === NDCG@k
        for(int k=1; k<=topK; k++)
        {
            float NDCG = NDCGSum[k]/UserNum_TestData;
            System.out.println("NDCG@"+Integer.toString(k)+":"+Float.toString(NDCG));
        }
        // === 1-call@k
        for(int k=1; k<=topK; k++)
        {
            float OneCall = OneCallSum[k]/UserNum_TestData;
            System.out.println("1-call@"+Integer.toString(k)+":"+Float.toString(OneCall));
        }
    }

    public static void testNDCG(HashMap<Integer, HashSet<Integer>> TestData)
    {
        // TestData: user->items
        // =========================================================
        float[] NDCGSum = new float[topK+1];

        // === calculate the best DCG, which can be used later
        float[] DCGbest = new float[topK+1];
        for (int k=1; k<=topK; k++)
        {
            DCGbest[k] = DCGbest[k-1];
            DCGbest[k] += 1/Math.log(k+1);
        }

        // === number of test cases
        int UserNum_TestData = TestData.keySet().size();

        for(int u=1; u<=n; u++)
        {
            // === check whether the user $u$ is in the test set
            if (!TestData.containsKey(u))
            {
                continue;
            }

            // ===
            HashSet<Integer> itemSet_u_TrainData = new HashSet<Integer>();
            if (TrainData.containsKey(u))
            {
                itemSet_u_TrainData = TrainData.get(u);
            }
            HashSet<Integer> itemSet_u_TestData = TestData.get(u);

            // =========================================================
            // === prediction
            HashMap<Integer, Float> item2Prediction = new HashMap<Integer, Float>();
            item2Prediction.clear();

            for(int i=1; i<=m; i++)
            {
                if ( !itemSetWhole.contains(i) || itemSet_u_TrainData.contains(i) )
                    continue;

                // === prediction via inner product
                float pred = biasV[i];
                for (int f=0; f<d; f++)
                {
                    pred += U[u][f]*V[i][f];
                }
                item2Prediction.put(i, pred);
            }
            // === sort
            List<Map.Entry<Integer,Float>> listY =
                    new ArrayList<Map.Entry<Integer,Float>>(item2Prediction.entrySet());
            Collections.sort(listY, new Comparator<Map.Entry<Integer,Float>>()
            {
                public int compare( Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2 )
                {
                    return o2.getValue().compareTo( o1.getValue() );
                }
            });

            // =========================================================
            // === Extract the topK recommended items
            int k=1;
            int[] TopKResult = new int [topK+1];
            Iterator<Entry<Integer, Float>> iter = listY.iterator();
            while (iter.hasNext())
            {
                if(k>topK)
                    break;

                Map.Entry<Integer, Float> entry = (Map.Entry<Integer, Float>) iter.next();
                int itemID = entry.getKey();
                TopKResult[k] = itemID;
                k++;
            }
            // === TopK evaluation
            float[] DCG = new float[topK+1];
            float[] DCGbest2 = new float[topK+1];
            for(k=1; k<=topK; k++)
            {
                // ===
                DCG[k] = DCG[k-1];
                int itemID = TopKResult[k];
                if ( itemSet_u_TestData.contains(itemID) )
                {
                    DCG[k] += 1 / Math.log(k+1);
                }
                // === in case the the number relevant items is smaller than k
                if (itemSet_u_TestData.size()>=k)
                {
                    DCGbest2[k] = DCGbest[k];
                }
                else
                {
                    DCGbest2[k] = DCGbest2[k-1];
                }
                NDCGSum[k] += DCG[k]/DCGbest2[k];
            }
        }

        // === NDCG@k
        int k = 5;
        float NDCG = NDCGSum[k]/UserNum_TestData;
        System.out.println("NDCG@"+Integer.toString(k)+":"+Float.toString(NDCG));
    }
}