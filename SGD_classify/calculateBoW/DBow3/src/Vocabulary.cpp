#include "Vocabulary.h"
#include "DescManip.h"
#include "quicklz.h"
#include <sstream>
#include "timers.h"

namespace DBoW3{
// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (int k, int L, WeightingType weighting, ScoringType scoring)
  : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
  m_scoring_object(NULL)
{
  createScoringObject();
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (const std::string &filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (const char *filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------


void Vocabulary::createScoringObject()
{
  delete m_scoring_object;
  m_scoring_object = NULL;

  switch(m_scoring)
  {
    case L1_NORM:
      m_scoring_object = new L1Scoring;
      break;

    case L2_NORM:
      m_scoring_object = new L2Scoring;
      break;

    case CHI_SQUARE:
      m_scoring_object = new ChiSquareScoring;
      break;

    case KL:
      m_scoring_object = new KLScoring;
      break;

    case BHATTACHARYYA:
      m_scoring_object = new BhattacharyyaScoring;
      break;

    case DOT_PRODUCT:
      m_scoring_object = new DotProductScoring;
      break;

  }
}

// --------------------------------------------------------------------------


void Vocabulary::setScoringType(ScoringType type)
{
  m_scoring = type;
  createScoringObject();
}

// --------------------------------------------------------------------------


void Vocabulary::setWeightingType(WeightingType type)
{
  this->m_weighting = type;
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary(
  const Vocabulary &voc)
  : m_scoring_object(NULL)
{
  *this = voc;
}

// --------------------------------------------------------------------------


Vocabulary::~Vocabulary()
{
  delete m_scoring_object;
}

// --------------------------------------------------------------------------


Vocabulary&
Vocabulary::operator=
  (const Vocabulary &voc)
{
  this->m_k = voc.m_k;
  this->m_L = voc.m_L;
  this->m_scoring = voc.m_scoring;
  this->m_weighting = voc.m_weighting;

  this->createScoringObject();

  this->m_nodes.clear();
  this->m_words.clear();

  this->m_nodes = voc.m_nodes;
  this->createWords();

  return *this;
}



void Vocabulary::create(
  const std::vector< cv::Mat > &training_features)
{
    std::vector<std::vector<cv::Mat> > vtf(training_features.size());
    for(size_t i=0;i<training_features.size();i++){
        vtf[i].resize(training_features[i].rows);
        for(int r=0;r<training_features[i].rows;r++)
            vtf[i][r]=training_features[i].rowRange(r,r+1);
    }
    create(vtf);

}

void Vocabulary::create(
  const std::vector<std::vector<cv::Mat> > &training_features)
{
  m_nodes.clear();
  m_words.clear();

  // expected_nodes = Sum_{i=0..L} ( k^i )
    int expected_nodes =
        (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));

  m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree


  std::vector<cv::Mat> features;
  getFeatures(training_features, features);


  // create root
  std::cout<<"create root"<<std::endl;
  m_nodes.push_back(Node(0)); // root

  // create the tree
  std::cout<<"create the tree"<<std::endl;
  HKmeansStep(0, features, 1);

  // create the words
  std::cout<<"create the words"<<std::endl;
  createWords();

  // and set the weight of each node of the tree
  std::cout<<"set the weight of each node of the tree"<<std::endl;
  setNodeWeights(training_features);

}

// --------------------------------------------------------------------------


void Vocabulary::create(
  const std::vector<std::vector<cv::Mat> > &training_features,
  int k, int L)
{
  m_k = k;
  m_L = L;

  create(training_features);
}

// --------------------------------------------------------------------------


void Vocabulary::create(
  const std::vector<std::vector<cv::Mat> > &training_features,
  int k, int L, WeightingType weighting, ScoringType scoring)
{
  m_k = k;
  m_L = L;
  m_weighting = weighting;
  m_scoring = scoring;
  createScoringObject();

  create(training_features);
}

// --------------------------------------------------------------------------


void Vocabulary::getFeatures(
  const std::vector<std::vector<cv::Mat> > &training_features,
  std::vector<cv::Mat> &features) const
{
  features.resize(0);
  for(size_t i=0;i<training_features.size();i++)
      for(size_t j=0;j<training_features[i].size();j++)
              features.push_back(training_features[i][j]);
}

// --------------------------------------------------------------------------


void Vocabulary::HKmeansStep(NodeId parent_id,
                             const std::vector<cv::Mat> &descriptors, int current_level)
{
    std::cout<<"start train: level =  "<<current_level<<std::endl;
    static int max_iter = 10000;
    int cur_iter = 0;

    if(descriptors.empty()) return;

    // features associated to each cluster
    std::vector<cv::Mat> clusters;
    std::vector<std::vector<unsigned int> > groups; // groups[i] = [j1, j2, ...]
    // j1, j2, ... indices of descriptors associated to cluster i

    clusters.reserve(m_k);
    groups.reserve(m_k);


    if((int)descriptors.size() <= m_k)
    {
        // trivial case: one cluster per feature
        groups.resize(descriptors.size());

        for(unsigned int i = 0; i < descriptors.size(); i++)
        {
            groups[i].push_back(i);
            clusters.push_back(descriptors[i]);
        }
    }
    else
    {
        // select clusters and groups with kmeans

        bool first_time = true;
        bool goon = true;

        // to check if clusters move after iterations
        std::vector<int> last_association, current_association;

        while(goon)
        {
            // 1. Calculate clusters

            if(first_time)
            {
                // random sample
//                std::cout<<"current_level---"<<current_level<<"---first_time"<<std::endl;
                initiateClusters(descriptors, clusters);
            }
            else
            {
                // calculate cluster centres

                for(unsigned int c = 0; c < clusters.size(); ++c)
                {
                    std::vector<cv::Mat> cluster_descriptors;
                    cluster_descriptors.reserve(groups[c].size());
                    std::vector<unsigned int>::const_iterator vit;
                    for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)
                    {
                        cluster_descriptors.push_back(descriptors[*vit]);
                    }

                    DescManip::meanValue(cluster_descriptors, clusters[c]);
                }

            } // if(!first_time)

            // 2. Associate features with clusters

            // calculate distances to cluster centers
            groups.clear();
            groups.resize(clusters.size(), std::vector<unsigned int>());
            current_association.resize(descriptors.size());

            //assoc.clear();

            //unsigned int d = 0;
            for(auto  fit = descriptors.begin(); fit != descriptors.end(); ++fit)//, ++d)
            {
                double best_dist = DescManip::distance((*fit), clusters[0]);
                unsigned int icluster = 0;

                for(unsigned int c = 1; c < clusters.size(); ++c)
                {
                    double dist = DescManip::distance((*fit), clusters[c]);
                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        icluster = c;
                    }
                }
                //assoc.ref<unsigned char>(icluster, d) = 1;

                groups[icluster].push_back(fit - descriptors.begin());
                current_association[ fit - descriptors.begin() ] = icluster;
            }

            // kmeans++ ensures all the clusters has any feature associated with them

            // 3. check convergence
            if(first_time)
            {
                first_time = false;
            }
            else
            {
                //goon = !eqUChar(last_assoc, assoc);

                goon = false;
                
                cur_iter++;
                
                if(cur_iter % 100 == 0){
                    std::cout<<"cur_iter = "<<cur_iter<<std::endl;
                }
                if(cur_iter > max_iter){
                    goon = false;
                    break;
                }
                
                for(unsigned int i = 0; i < current_association.size(); i++)
                {
                    if(current_association[i] != last_association[i]){
                        goon = true;
                        break;
                    }
                }
            }

            if(goon)
            {
                // copy last feature-cluster association
                last_association = current_association;
                //last_assoc = assoc.clone();
            }

        } // while(goon)

    } // if must run kmeans
    std::cout<<"use iter = "<<cur_iter<<std::endl;
    // create nodes
    for(unsigned int i = 0; i < clusters.size(); ++i)
    {
        NodeId id = m_nodes.size();
        m_nodes.push_back(Node(id));
        m_nodes.back().descriptor = clusters[i];
        m_nodes.back().parent = parent_id;
        m_nodes[parent_id].children.push_back(id);
    }

//    std::cout<<"m_nodes.size()---"<<clusters.size()<<std::endl;
    
    // go on with the next level
    if(current_level < m_L)
    {
        // iterate again with the resulting clusters
        const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;
        for(unsigned int i = 0; i < clusters.size(); ++i)
        {
            NodeId id = children_ids[i];

            std::vector<cv::Mat> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for(vit = groups[i].begin(); vit != groups[i].end(); ++vit)
            {
                child_features.push_back(descriptors[*vit]);
            }
            if(child_features.size() > 1)
            {
                double current_dist = DescManip::distance(child_features[0], child_features[1]);
                std::cout<<current_dist<<std::endl;
            }

            if(child_features.size() > 1)
            {
                HKmeansStep(id, child_features, current_level + 1);
            }
        }
    }
    
    
}

// --------------------------------------------------------------------------


void Vocabulary::initiateClusters
  (const std::vector<cv::Mat> &descriptors,
   std::vector<cv::Mat> &clusters) const
{
  initiateClustersKMpp(descriptors, clusters);
}

// --------------------------------------------------------------------------


void Vocabulary::initiateClustersKMpp(
  const std::vector<cv::Mat> &pfeatures,
    std::vector<cv::Mat> &clusters) const
{
  // Implements kmeans++ seeding algorithm
  // Algorithm:
  // 1. Choose one center uniformly at random from among the data points.
  // 2. For each data point x, compute D(x), the distance between x and the nearest
  //    center that has already been chosen.
  // 3. Add one new data point as a center. Each point x is chosen with probability
  //    proportional to D(x)^2.
  // 4. Repeat Steps 2 and 3 until k centers have been chosen.
  // 5. Now that the initial centers have been chosen, proceed using standard k-means
  //    clustering.


//  DUtils::Random::SeedRandOnce();

  clusters.resize(0);
  clusters.reserve(m_k);
  std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

  // 1.

  int ifeature = rand()% pfeatures.size();//DUtils::Random::RandomInt(0, pfeatures.size()-1);

  // create first cluster
  clusters.push_back(pfeatures[ifeature]);

  // compute the initial distances
   std::vector<double>::iterator dit;
  dit = min_dists.begin();
  for(auto fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
  {
    *dit = DescManip::distance((*fit), clusters.back());
  }

  while((int)clusters.size() < m_k)
  {
    // 2.
    dit = min_dists.begin();
    for(auto  fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
    {
      if(*dit > 0)
      {
        double dist = DescManip::distance((*fit), clusters.back());
        if(dist < *dit) *dit = dist;
      }
    }

    // 3.
    double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

    if(dist_sum > 0)
    {
      double cut_d;
      do
      {

        cut_d = (double(rand())/ double(RAND_MAX))* dist_sum;
      } while(cut_d == 0.0);

      double d_up_now = 0;
      for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
      {
        d_up_now += *dit;
        if(d_up_now >= cut_d) break;
      }

      if(dit == min_dists.end())
        ifeature = pfeatures.size()-1;
      else
        ifeature = dit - min_dists.begin();


      clusters.push_back(pfeatures[ifeature]);
    } // if dist_sum > 0
    else
      break;

  } // while(used_clusters < m_k)

}

// --------------------------------------------------------------------------


void Vocabulary::createWords()
{
  m_words.resize(0);

  if(!m_nodes.empty())
  {
    m_words.reserve( (int)pow((double)m_k, (double)m_L) );


    auto  nit = m_nodes.begin(); // ignore root
    for(++nit; nit != m_nodes.end(); ++nit)
    {
      if(nit->isLeaf())
      {
        nit->word_id = m_words.size();
        m_words.push_back( &(*nit) );
      }
    }
  }
}

// --------------------------------------------------------------------------


void Vocabulary::setNodeWeights
  (const std::vector<std::vector<cv::Mat> > &training_features)
{
  const unsigned int NWords = m_words.size();
  const unsigned int NDocs = training_features.size();

  if(m_weighting == TF || m_weighting == BINARY)
  {
    // idf part must be 1 always
    for(unsigned int i = 0; i < NWords; i++)
      m_words[i]->weight = 1;
  }
  else if(m_weighting == IDF || m_weighting == TF_IDF)
  {
    // IDF and TF-IDF: we calculte the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform

    std::vector<unsigned int> Ni(NWords, 0);
    std::vector<bool> counted(NWords, false);


    for(auto mit = training_features.begin(); mit != training_features.end(); ++mit)
    {
      fill(counted.begin(), counted.end(), false);

      for(auto fit = mit->begin(); fit < mit->end(); ++fit)
      {
        WordId word_id;
        transform(*fit, word_id);

        if(!counted[word_id])
        {
          Ni[word_id]++;
          counted[word_id] = true;
        }
      }
    }

    // set ln(N/Ni)
    for(unsigned int i = 0; i < NWords; i++)
    {
      if(Ni[i] > 0)
      {
        m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
      }// else // This cannot occur if using kmeans++
    }

  }

}

// --------------------------------------------------------------------------






// --------------------------------------------------------------------------


float Vocabulary::getEffectiveLevels() const
{
  long sum = 0;
   for(auto wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    const Node *p = *wit;

    for(; p->id != 0; sum++) p = &m_nodes[p->parent];
  }

  return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------


cv::Mat Vocabulary::getWord(WordId wid) const
{
  return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------


WordValue Vocabulary::getWordWeight(WordId wid) const
{
  return m_words[wid]->weight;
}

// --------------------------------------------------------------------------


WordId Vocabulary::transform
  (const cv::Mat& feature) const
{
  if(empty())
  {
    return 0;
  }

  WordId wid;
  transform(feature, wid);
  return wid;
}

// --------------------------------------------------------------------------

void Vocabulary::transform(
        const cv::Mat& features, BowVector &v) const
{
    //    std::vector<cv::Mat> vf(features.rows);
    //    for(int r=0;r<features.rows;r++) vf[r]=features.rowRange(r,r+1);
    //    transform(vf,v);



    v.clear();

    if(empty())
    {
        return;
    }

    // normalize
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);


    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        for(int r=0;r<features.rows;r++)
        {
            WordId id;
            WordValue w;
            // w is the idf value if TF_IDF, 1 if TF
            transform(features.row(r), id, w);
            // not stopped
            if(w > 0){
                v.addWeight(id, w);
            }
        }

        if(!v.empty() && !must)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                vit->second /= nd;
        }

    }
    else // IDF || BINARY
    {
        for(int r=0;r<features.rows;r++)
        {
            WordId id;
            WordValue w;
            // w is idf if IDF, or 1 if BINARY

            transform(features.row(r), id, w);

            // not stopped
            if(w > 0) v.addIfNotExist(id, w);

        } // if add_features
    } // if m_weighting == ...

    if(must) v.normalize(norm);

}



void Vocabulary::transform(
  const std::vector<cv::Mat>& features, BowVector &v) const
{
  v.clear();

  if(empty())
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);


  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    for(auto fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w);

      // not stopped
      if(w > 0) v.addWeight(id, w);
    }

    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

  }
  else // IDF || BINARY
  {
    for(auto fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w);

      // not stopped
      if(w > 0) v.addIfNotExist(id, w);

    } // if add_features
  } // if m_weighting == ...

  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------
void Vocabulary::transform_vec(const std::vector<cv::Mat>& features,std::vector<BowVector> &desc_vec_bow, std::vector<FeatureVector> &desc_vec_fet, int levelsup ,int vec_number) const
{
    desc_vec_bow.clear();
    desc_vec_fet.clear();
    
    if(empty()) // safe for subclasses
    {
        return;
    }
    
    // normalize
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);
    
    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        unsigned int i_feature = 0;
        for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
        {
            std::vector<WordId> word_id_vec;
            std::vector<NodeId> node_id_vec;
            std::vector<WordValue> word_value_vec;
            // w is the idf value if TF_IDF, 1 if TF
            
            transform_vec2(*fit, word_id_vec, word_value_vec, node_id_vec, levelsup,vec_number);
            for (int i=0; i<vec_number; i++)
            {
                if(word_value_vec[i] > 0) // not stopped
                {
                    BowVector aaa;  aaa.addWeight(word_id_vec[i], word_value_vec[i]);
                    desc_vec_bow.push_back(aaa);
                    FeatureVector bbb;bbb.addFeature(node_id_vec[i], i_feature);
                    desc_vec_fet.push_back(bbb);
                }
            }

        }
        for (BowVector& desc_bow:desc_vec_bow)
        {
            if(!desc_bow.empty() && !must)
            {
                // unnecessary when normalizing
                const double nd = desc_bow.size();
                for(BowVector::iterator vit = desc_bow.begin(); vit != desc_bow.end(); vit++)
                    vit->second /= nd;
            }
        }
        
    }
    else // IDF || BINARY
    {
        unsigned int i_feature = 0;
        for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
        {
            std::vector<WordId> id;
            std::vector<NodeId> nid;
            std::vector<WordValue> w;
            // w is the idf value if TF_IDF, 1 if TF
            
            transform_vec2(*fit, id, w, nid, levelsup,vec_number);
            for (int i=0; i<vec_number; i++)
            {
                if(w[i] > 0) // not stopped
                {
                    BowVector aaa;  aaa.addWeight(id[i], w[i]);
                    desc_vec_bow.push_back(aaa);
                    FeatureVector bbb;bbb.addFeature(nid[i], i_feature);
                    desc_vec_fet.push_back(bbb);
                }
            }
        }
    } // if m_weighting == ...
    
    if(must)
        for (BowVector& desc_bow:desc_vec_bow)
            desc_bow.normalize(norm);
}

void Vocabulary::transform(
  const std::vector<cv::Mat>& features,
  BowVector &v, FeatureVector &fv, int levelsup) const
{
  v.clear();
  fv.clear();

  if(empty()) // safe for subclasses
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);


  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    unsigned int i_feature = 0;
    for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w, &nid, levelsup);

      if(w > 0) // not stopped
      {
        v.addWeight(id, w);
        fv.addFeature(nid, i_feature);
      }
    }

    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

  }
  else // IDF || BINARY
  {
    unsigned int i_feature = 0;
    for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w, &nid, levelsup);

      if(w > 0) // not stopped
      {
        v.addIfNotExist(id, w);
        fv.addFeature(nid, i_feature);
      }
    }
  } // if m_weighting == ...

  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------


// --------------------------------------------------------------------------


void Vocabulary::transform
  (const cv::Mat &feature, WordId &id) const
{
  WordValue weight;
  transform(feature, id, weight);
}

// --------------------------------------------------------------------------


void Vocabulary::transform(const cv::Mat &feature,
  WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
{
  // propagate the feature down the tree


  // level at which the node must be stored in nid, if given
  const int nid_level = m_L - levelsup;
  if(nid_level <= 0 && nid != NULL) *nid = 0; // root

  NodeId final_id = 0; // root
  int current_level = 0;

  do
  {
    ++current_level;
    auto const  &nodes = m_nodes[final_id].children;
    double best_d = std::numeric_limits<double>::max();
//    DescManip::distance(feature, m_nodes[final_id].descriptor);

    for(const auto  &id:nodes)
    {
      double d = DescManip::distance(feature, m_nodes[id].descriptor);
      if(d < best_d)
      {
        best_d = d;
        final_id = id;
      }
    }

    if(nid != NULL && current_level == nid_level)
    {
      *nid = final_id;
        weight =1;
        return;   //we can return in early level if we don't care the value of weight. Warning: only available if weight is NOTT NEEDED
    }
  } while( !m_nodes[final_id].isLeaf() );

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}
     
void Vocabulary::transform_vec2(const cv::Mat &feature,
                               std::vector<WordId> &word_id_vec, std::vector<WordValue> &weight_vec, std::vector<NodeId> &nid_vec, int levelsup, int vec_number) const
{
    // level at which the node must be stored in nid, if given
    const int nid_level = m_L - levelsup;
    for (int i=0; i< vec_number; i++)
    {
        if(nid_level <= 0 && nid_vec[i] != 0)
            nid_vec.push_back(0); // root
    }

    NodeId final_id = 0; // root
    int current_level = 0;
    double best_d;
    std::map<double,DBoW3::NodeId>  dist_map;
    dist_map.clear();
    int i=0;
    do
    {
        ++current_level;
        auto const  &nodes = m_nodes[final_id].children;
        best_d = std::numeric_limits<double>::max();
        for(const auto  &id:nodes)
        {
            double dist = DescManip::distance(feature, m_nodes[id].descriptor);
            std::cout<<"dist::"<<dist<<std::endl;
            std::cout<<"i-----------------------------------::"<<i++<<std::endl;
            if(dist<best_d)
            {
                best_d = dist;
                dist_map.insert(std::make_pair(dist,id));
                final_id = id;
                std::cout<<"best_d:"<<best_d<<std::endl;
            }
  
//            dist_map[id]=dist;
//            if(dist < best_d)
//            {
//                best_d = dist;
//                final_id = id;
//            }
        }
//        std::cout<<current_level<<std::endl;
        for (int i=0; i<vec_number; i++) {
            if(current_level == nid_level)
                nid_vec.push_back(final_id);
        }

        
    } while( !m_nodes[final_id].isLeaf() );
    
    
    // turn node id into word id
    std::map<double,DBoW3::NodeId>::iterator it=dist_map.begin();
    for(int i=0;i<vec_number;i++,it++)
    {
        word_id_vec.push_back(m_nodes[it->second].word_id);
        std::cout<<it->first<<std::endl;
        std::cout<<it->second<<std::endl;
        weight_vec.push_back(m_nodes[it->second].weight);
    }
}
void Vocabulary::transform_min_distance(const cv::Mat &feature,
  WordId &word_id1, double &distance1,WordId &word_id2, double &distance2 ) const
{
    // level at which the node must be stored in nid, if given
    NodeId final_id = 0; // root
    NodeId best_id = 0,second_id = 0; // root
    double best_d,sceond_d;
    //maximum speed by computing here distance and avoid calling to DescManip::distance
    do
    {
        auto const  &nodes = m_nodes[final_id].children;
        sceond_d = best_d = std::numeric_limits<double>::max();
        for(const auto  &id:nodes)
        {
            //compute distance
            double dist= DescManip::distance(feature, m_nodes[id].descriptor);
            if(dist <sceond_d)
            {
                sceond_d = dist;
                second_id = id;
                if(dist < best_d)
                {
                    sceond_d = best_d;
                    second_id = best_id;
                    best_d = dist;
                    best_id = id;
                    final_id = id;
                }
                
            }
        }
        
        
    } while( !m_nodes[final_id].isLeaf() );
// turn node id into word id
    word_id1 = m_nodes[best_id].word_id;
    distance1 = best_d;
    word_id2 = m_nodes[second_id].word_id;
    distance2 = sceond_d;
}
//void Vocabulary::transform_min_distance3(const std::vector<cv::Mat>& features,
//  std::vector<DBoW3::BowVector>& vec_bow, std::vector<DBoW3::FeatureVector>& vec_fet,int levelsup) const
//{
//    
//    if(levelsup>5 | levelsup<0)
//    {
//        return;
//    }
//    
//    BowVector v1,v2,v3;
//    FeatureVector fv1,fv2,fv3;
//    
//    if(empty()) // safe for subclasses
//    {
//        return;
//    }
//    
//    // normalize
//    LNorm norm;
//    bool must = m_scoring_object->mustNormalize(norm);
//    
//    
//    if(m_weighting == TF || m_weighting == TF_IDF)
//    {
//        unsigned int i_feature = 0;
//        for(int i=0;i<features.size();i++, ++i_feature)
//        {
//            cv::Mat fit = features[i];
//            WordId id1,id2,id3;
//            NodeId nid1,nid2,nid3;
//            WordValue w1,w2,w3;
//            // w is the idf value if TF_IDF, 1 if TF
//            
//            {
//                const int nid_level = m_L - levelsup;
//                NodeId final_id = 0; // root
//                int current_level = 0;
//                NodeId best_id,second_id,third_id = 0; // root
//                double best_d,sceond_d,third_d;
//                do
//                {
//                    ++current_level;
//                    auto const  &nodes = m_nodes[final_id].children;
//                    third_d = sceond_d = best_d = std::numeric_limits<double>::max();
//                    //    DescManip::distance(feature, m_nodes[final_id].descriptor);
//                    
//                    for(const auto  &id:nodes)
//                    {
//                        double dist = DescManip::distance(fit, m_nodes[id].descriptor);
//                        if(dist < third_d)
//                        {
//                            third_d = dist;
//                            third_id = id;
//                            if(dist <sceond_d)
//                            {
//                                third_d = sceond_d;
//                                third_id = second_id;
//                                sceond_d = dist;
//                                second_id = id;
//                                if(dist < best_d)
//                                {
//                                    sceond_d = best_d;
//                                    second_id = best_id;
//                                    best_d = dist;
//                                    best_id = id;
//                                    final_id = id;
//                                }
//                            }
//                        }
//                    }
//                    
//                    if(current_level == nid_level)
//                    {
//                        nid1 = best_id;
//                        nid2 = second_id;
//                        nid3 = third_id;
//                    }
//                    
//                } while( !m_nodes[final_id].isLeaf() );
//                
//                // turn node id into word id
//                id1 = m_nodes[best_id].word_id;
//                w1 = m_nodes[best_id].weight;
//                id2 = m_nodes[second_id].word_id;
//                w2 = m_nodes[second_id].weight;
//                id3 = m_nodes[third_id].word_id;
//                w3 = m_nodes[third_id].weight;
//            }
//            
//            if(w1 > 0) // not stopped
//            {
//                v1.addWeight(id1, w1);
//                fv1.addFeature(nid1, i_feature);
//            }
//            if(w2 > 0) // not stopped
//            {
//                v2.addWeight(id2, w2);
//                fv2.addFeature(nid2, i_feature);
//            }
//            if(w3 > 0) // not stopped
//            {
//                v3.addWeight(id3, w3);
//                fv3.addFeature(nid3, i_feature);
//            }
//        }
//        
//        if(!v1.empty() && !must)
//        {
//            // unnecessary when normalizing
//            const double nd = v1.size();
//            for(BowVector::iterator vit = v1.begin(); vit != v1.end(); vit++)
//                vit->second /= nd;
//        }
//        if(!v2.empty() && !must)
//        {
//            // unnecessary when normalizing
//            const double nd = v2.size();
//            for(BowVector::iterator vit = v2.begin(); vit != v2.end(); vit++)
//                vit->second /= nd;
//        }
//        if(!v3.empty() && !must)
//        {
//            // unnecessary when normalizing
//            const double nd = v3.size();
//            for(BowVector::iterator vit = v3.begin(); vit != v3.end(); vit++)
//                vit->second /= nd;
//        }
//        
//    }
//    
//    vec_bow.push_back(v1);
//    vec_bow.push_back(v2);
//    vec_bow.push_back(v3);
//    
//    vec_fet.push_back(fv1);
//    vec_fet.push_back(fv2);
//    vec_fet.push_back(fv3);
//}

void Vocabulary::transform_min_distance3(const std::vector<cv::Mat>& features,
                                         std::vector<DBoW3::BowVector>& vec_bow, std::vector<DBoW3::FeatureVector>& vec_fet,
                                         int levelsup,int number ) const
{
    if(levelsup>m_k || levelsup<0)
    {
        std::cout<<"levelsup is too large or small"<<std::endl;
        return;
    }
    std::vector<DBoW3::BowVector> vec_bow_temp(number);
    std::vector<DBoW3::FeatureVector> vec_fet_temp(number);
    
    if(empty()) // safe for subclasses
    {
        return;
    }
    
    // normalize
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);
    
    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        unsigned int i_feature = 0;
        for(std::size_t i=0;i<features.size();i++, ++i_feature)
        {
            cv::Mat fit = features[i];
            std::vector<WordId> id(number);
            std::vector<NodeId> nid(number);
            std::vector<WordValue> wordv(number);
            // w is the idf value if TF_IDF, 1 if TF
            NodeId final_id = 0; // root
            int current_level = 0;
            double best_d;
            do
            {
                std::map<double,NodeId> dist_map;
                ++current_level;
                auto const  &nodes = m_nodes[final_id].children;
                best_d = std::numeric_limits<double>::max();
                //    DescManip::distance(feature, m_nodes[final_id].descriptor);
                
                for(const auto  &id:nodes)
                {
                    double dist = DescManip::distance(fit, m_nodes[id].descriptor);
                    dist_map.insert(std::make_pair(dist,id));
                    if(dist < best_d)
                    {
                        best_d = dist;
                        final_id = id;
                    }
                }
                
                if(current_level == levelsup)
                {
                    int i =0;
                    for( std::map<double,NodeId>::iterator iter=dist_map.begin();iter!=dist_map.end()&& i<number ;iter++)
                        nid[i++] = iter->second;
                }
                
            } while( !m_nodes[final_id].isLeaf() && current_level != levelsup);
            
            // turn node id into word id
            for(int i=0;i<number;i++)
            {
                id[i]=(m_nodes[nid[i]].word_id);
                wordv[i]=(m_nodes[nid[i]].weight);
            }
            for(int i=0;i<number;i++)
            {
                if(wordv[i]>0)
                {
                    vec_bow_temp[i].addWeight(id[i], wordv[i]);
                    vec_fet_temp[i].addFeature(nid[i], i_feature);
                }
                
            }
        }
        for(int i=0;i<number;i++)
        {
            if(!vec_bow_temp[i].empty() && !must)
            {
                // unnecessary when normalizing
                const double nd = vec_bow_temp[i].size();
                for(BowVector::iterator vit = vec_bow_temp[i].begin(); vit != vec_bow_temp[i].end(); vit++)
                    vit->second /= nd;
            }
        }
    }
    vec_bow.swap(vec_bow_temp);
    vec_fet.swap(vec_fet_temp);
}

void Vocabulary::transform(const cv::Mat &feature,
                           WordId &word_id, WordValue &weight) const
{
    //propagate the feature down the tree
    std::vector<NodeId> nodes;
    int current_level = 0;
    
    //level at which the node must be stored in nid, if given
    NodeId final_id = 0; // root
    //maximum speed by computing here distance and avoid calling to DescManip::distance
    
    //binary descriptor
//    int ntimes=0;
    //  if (feature.type()==CV_8U)
    //  {
    //      do
    //      {
    //          auto const  &nodes = m_nodes[final_id].children;
    //          uint64_t best_d = std::numeric_limits<uint64_t>::max();
    //          int idx=0,bestidx=0;
    //           for(const auto  &id:nodes)
    //          {
    //              //compute distance
    //             //  std::cout<<idx<< " "<<id<<" "<< m_nodes[id].descriptor<<std::endl;
    //              uint64_t dist= DescManip::distance_8uc1(feature, m_nodes[id].descriptor);
    //              if(dist < best_d)
    //              {
    //                  best_d = dist;
    //                  final_id = id;
    //                  bestidx=idx;
    //              }
    //              idx++;
    //          }
    //        // std::cout<<bestidx<<" "<<final_id<<" d:"<<best_d<<" "<<m_nodes[final_id].descriptor<<  std::endl<<std::endl;
    //      } while( !m_nodes[final_id].isLeaf() );
    //   }
    //
    ////      uint64_t ret=0;
    ////      const uchar *pb = b.ptr<uchar>();
    ////      for(int i=0;i<a.cols;i++,pa++,pb++){
    ////          uchar v=(*pa)^(*pb);
    ////#ifdef __GNUG__
    ////          ret+=__builtin_popcount(v);//only in g++
    ////#else
    //
    ////          ret+=v& (1<<0);
    ////          ret+=v& (1<<1);
    ////          ret+=v& (1<<2);
    ////          ret+=v& (1<<3);
    ////          ret+=v& (1<<4);
    ////          ret+=v& (1<<5);
    ////          ret+=v& (1<<6);
    ////          ret+=v& (1<<7);
    ////#endif
    ////  }
    ////      return ret;
    ////  }
    ////  else{
    ////      double sqd = 0.;
    ////      assert(a.type()==CV_32F);
    ////      assert(a.rows==1);
    ////      const float *a_ptr=a.ptr<float>(0);
    ////      const float *b_ptr=b.ptr<float>(0);
    ////      for(int i = 0; i < a.cols; i ++)
    ////          sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
    ////      return sqd;
    ////  }
    //
    //
    ////  do
    ////  {
    ////    auto const  &nodes = m_nodes[final_id].children;
    ////    double best_d = std::numeric_limits<double>::max();
    //
    ////    for(const auto  &id:nodes)
    ////    {
    ////      double d = DescManip::distance(feature, m_nodes[id].descriptor);
    ////      if(d < best_d)
    ////      {
    ////        best_d = d;
    ////        final_id = id;
    ////      }
    ////    }
    ////  } while( !m_nodes[final_id].isLeaf() );
    //
    //  // turn node id into word id
    ////  word_id = m_nodes[final_id].word_id;
    ////  weight = m_nodes[final_id].weight;
    //
    //    else
    {
        do
        {
            ++current_level;
            nodes = m_nodes[final_id].children;
            final_id = nodes[0];
            
            double best_d = DescManip::distance(feature, m_nodes[final_id].descriptor);
            
            for(auto nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
            {
                NodeId id = *nit;
                double d = DescManip::distance(feature, m_nodes[id].descriptor);
                if(d < best_d)
                {
                    best_d = d;
                    final_id = id;
                }
            }
            
            
        } while( !m_nodes[final_id].isLeaf() );
    }
    
    // turn node id into word id
    word_id = m_nodes[final_id].word_id;
    weight = m_nodes[final_id].weight;
}
// --------------------------------------------------------------------------

NodeId Vocabulary::getParentNode
  (WordId wid, int levelsup) const
{
  NodeId ret = m_words[wid]->id; // node id
  while(levelsup > 0 && ret != 0) // ret == 0 --> root
  {
    --levelsup;
    ret = m_nodes[ret].parent;
  }
  return ret;
}

// --------------------------------------------------------------------------


void Vocabulary::getWordsFromNode
  (NodeId nid, std::vector<WordId> &words) const
{
  words.clear();

  if(m_nodes[nid].isLeaf())
  {
    words.push_back(m_nodes[nid].word_id);
  }
  else
  {
    words.reserve(m_k); // ^1, ^2, ...

    std::vector<NodeId> parents;
    parents.push_back(nid);

    while(!parents.empty())
    {
      NodeId parentid = parents.back();
      parents.pop_back();

      const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
      std::vector<NodeId>::const_iterator cit;

      for(cit = child_ids.begin(); cit != child_ids.end(); ++cit)
      {
        const Node &child_node = m_nodes[*cit];

        if(child_node.isLeaf())
          words.push_back(child_node.word_id);
        else
          parents.push_back(*cit);

      } // for each child
    } // while !parents.empty
  }
}

// --------------------------------------------------------------------------


int Vocabulary::stopWords(double minWeight)
{
  int c = 0;
   for(auto wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    if((*wit)->weight < minWeight)
    {
      ++c;
      (*wit)->weight = 0;
    }
  }
  return c;
}

// --------------------------------------------------------------------------


//void Vocabulary::save(const std::vector<std::vector<cv::Mat> > &training_features,const std::string &filename,  bool binary_compressed) const
    void Vocabulary::save(const std::string &filename,  bool binary_compressed) const
{

    if ( filename.find(".yml")==std::string::npos){
        std::ofstream file_out(filename);
        if (!file_out) throw std::runtime_error("Vocabulary::saveBinary Could not open file :"+filename+" for writing");
        toStream(file_out,binary_compressed);
    }
    else{
        cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
        if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
//        save(training_features,fs);
        save(fs);
    }
}

// --------------------------------------------------------------------------


bool Vocabulary::load(const std::string &filename)
{
    //check first if it is a binary file
    std::ifstream ifile(filename);
    if (!ifile)
    {
        std::runtime_error("Vocabulary::load Could not open file :"+filename+" for reading");
        return false;
    }
    uint64_t sig;//magic number describing the file
    ifile.read((char*)&sig,sizeof(sig));
    if (sig==88877711233) {//it is a binary file. read from it
        ifile.seekg(0,std::ios::beg);
        fromStream(ifile);

    }
    else{
        if ( filename.find(".txt")!=std::string::npos){
            //read from a text file (used in ORBSLAM2)
            load_fromtxt(filename);
        }
        else{
        cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
        if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
        this->load(fs);
        }
    }
    return true;
}


//void Vocabulary::save(const std::vector<std::vector<cv::Mat> > &training_features,cv::FileStorage &f,
//  const std::string &name) const
void Vocabulary::save(cv::FileStorage &f,
                      const std::string &name) const
{

  f << name << "{";

  f << "k" << m_k;
  f << "L" << m_L;
  f << "scoringType" << m_scoring;
  f << "weightingType" << m_weighting;

  // tree
  f << "nodes" << "[";
  std::vector<NodeId> parents, children;
  std::vector<NodeId>::const_iterator pit;

  parents.push_back(0); // root

  while(!parents.empty())
  {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node& parent = m_nodes[pid];
    children = parent.children;

    for(pit = children.begin(); pit != children.end(); pit++)
    {
      const Node& child = m_nodes[*pit];
//      std::cout<<m_nodes[*pit].id<<" ";

      // save node data
      f << "{:";
      f << "nodeId" << (int)child.id;
      f << "parentId" << (int)pid;
      f << "weight" << (double)child.weight;
      f << "descriptor" << DescManip::toString(child.descriptor);
      f << "}";

      // add to parent list
      if(!child.isLeaf())
      {
        parents.push_back(*pit);
      }
    }
  }
  std::cout<<"\n";

  f << "]"; // nodes

  // words
  f << "words" << "[";

   for(auto wit = m_words.begin(); wit != m_words.end(); wit++)
  {
    WordId id = wit - m_words.begin();
    f << "{:";
    f << "wordId" << (int)id;
    f << "nodeId" << (int)(*wit)->id;
    f << "}";
  }

  f << "]"; // words

  f << "}";
    
//    f<<"Bow_vector";
//    f<<"Feature_vector";
//    DBoW3::BowVector b_vec;
//    DBoW3::FeatureVector f_vec;
//    for ( int i=0; i<training_features.size(); i++ )
//    {
//        std::cout<<training_features[i][0]<<std::endl;
////        m_voc->transform(training_features[i], b_vec);
//        std::cout<<b_vec<<std::endl;
//        m_voc->transform(training_features[i], b_vec, f_vec,1);
//        std::cout<<b_vec<<std::endl;
//        std::cout<<f_vec<<std::endl;
//        f << "}";
//    }

    

}

void Vocabulary::toStream(  std::ostream &out_str, bool compressed) const throw(std::exception){

    uint64_t sig=88877711233;//magic number describing the file
    out_str.write((char*)&sig,sizeof(sig));
    out_str.write((char*)&compressed,sizeof(compressed));
    uint32_t nnodes=m_nodes.size();
    out_str.write((char*)&nnodes,sizeof(nnodes));
    if (nnodes==0)return;
    //save everything to a stream
    std::stringstream aux_stream;
    aux_stream.write((char*)&m_k,sizeof(m_k));
    aux_stream.write((char*)&m_L,sizeof(m_L));
    aux_stream.write((char*)&m_scoring,sizeof(m_scoring));
    aux_stream.write((char*)&m_weighting,sizeof(m_weighting));
    //nodes
    std::vector<NodeId> parents={0};// root


    while(!parents.empty())
    {
        NodeId pid = parents.back();
        parents.pop_back();

        const Node& parent = m_nodes[pid];

        for(auto pit :parent.children)
        {

            const Node& child = m_nodes[pit];
            aux_stream.write((char*)&child.id,sizeof(child.id));
            aux_stream.write((char*)&pid,sizeof(pid));
            aux_stream.write((char*)&child.weight,sizeof(child.weight));
            DescManip::toStream(child.descriptor,aux_stream);
            // add to parent list
            if(!child.isLeaf()) parents.push_back(pit);
        }
    }
    //words
    //save size
    uint32_t m_words_size=m_words.size();
    aux_stream.write((char*)&m_words_size,sizeof(m_words_size));
    for(auto wit = m_words.begin(); wit != m_words.end(); wit++)
    {
        WordId id = wit - m_words.begin();
        aux_stream.write((char*)&id,sizeof(id));
        aux_stream.write((char*)&(*wit)->id,sizeof((*wit)->id));
    }


    //now, decide if compress or not
    if (compressed){
        qlz_state_compress  state_compress;
        memset(&state_compress, 0, sizeof(qlz_state_compress));
        //Create output buffer
        int chunkSize=10000;
        std::vector<char> compressed( chunkSize+size_t(400), 0);
        std::vector<char> input( chunkSize, 0);
        auto total_size= aux_stream.tellp();
        uint64_t total_compress_size=0;
        //calculate how many chunks will be written
        uint32_t nChunks= total_size/chunkSize;
        if ( total_size%chunkSize!=0) nChunks++;
        out_str.write((char*)&nChunks, sizeof(nChunks));
        //start compressing the chunks
        while(total_size!=0){
            int readSize=chunkSize;
            if (total_size<chunkSize) readSize=total_size;
            aux_stream.read(&input[0],readSize);
            uint64_t  compressed_size   = qlz_compress(&input[0], &compressed[0], readSize, &state_compress);
            total_size-=readSize;
            out_str.write(&compressed[0], compressed_size);
            total_compress_size+=compressed_size;
        }
    }
    else{
        out_str<<aux_stream.rdbuf();
    }
}


void Vocabulary:: load_fromtxt(const std::string &filename)throw(std::runtime_error){

    std::ifstream ifile(filename);
    if(!ifile)throw std::runtime_error("Vocabulary:: load_fromtxt  Could not open file for reading:"+filename);
    int n1, n2;
    {
    std::string str;
    getline(ifile,str);
    std::stringstream ss(str);
    ss>>m_k>>m_L>>n1>>n2;
    }
    if(m_k<0 || m_k>20 || m_L<1 || m_L>10 || n1<0 || n1>5 || n2<0 || n2>3)
         throw std::runtime_error( "Vocabulary loading failure: This is not a correct text file!" );

    m_scoring = (ScoringType)n1;
    m_weighting = (WeightingType)n2;
    createScoringObject();
    // nodes
       int expected_nodes =
       (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));
       m_nodes.reserve(expected_nodes);

       m_words.reserve(pow((double)m_k, (double)m_L + 1));

       m_nodes.resize(1);
       m_nodes[0].id = 0;

       int counter=0;
       while(!ifile.eof()){
           std::string snode;
           getline(ifile,snode);
           if (counter++%100==0)std::cerr<<".";
          // std::cout<<snode<<std::endl;
           if (snode.size()==0)break;
           std::stringstream ssnode(snode);

           int nid = m_nodes.size();
           m_nodes.resize(m_nodes.size()+1);
           m_nodes[nid].id = nid;

           int pid ;
           ssnode >> pid;
           m_nodes[nid].parent = pid;
           m_nodes[pid].children.push_back(nid);

           int nIsLeaf;
           ssnode >> nIsLeaf;

           //read until the end and add to data
           std::vector<float> data;data.reserve(100);
           float d;
           while( ssnode>>d) data.push_back(d);
           //the weight is the last
           m_nodes[nid].weight=data.back();
           data.pop_back();//remove
           //the rest, to the descriptor
           m_nodes[nid].descriptor.create(1,data.size(),CV_8UC1);
           auto ptr=m_nodes[nid].descriptor.ptr<uchar>(0);
           for(auto d:data) *ptr++=d;


           if(nIsLeaf>0)
           {
               int wid = m_words.size();
               m_words.resize(wid+1);

               m_nodes[nid].word_id = wid;
               m_words[wid] = &m_nodes[nid];
           }
           else
           {
               m_nodes[nid].children.reserve(m_k);
           }
       }
}
void Vocabulary::fromStream(  std::istream &str )   throw(std::exception){


    m_words.clear();
    m_nodes.clear();
    uint64_t sig=0;//magic number describing the file
    str.read((char*)&sig,sizeof(sig));
    if (sig!=88877711233) throw std::runtime_error("Vocabulary::fromStream  is not of appropriate type");
    bool compressed;
    str.read((char*)&compressed,sizeof(compressed));
    uint32_t nnodes;
    str.read((char*)&nnodes,sizeof(nnodes));
    if(nnodes==0)return;
    std::stringstream decompressed_stream;
    std::istream *_used_str=0;
    if (compressed){
        qlz_state_decompress state_decompress;
        memset(&state_decompress, 0, sizeof(qlz_state_decompress));
        int chunkSize=10000;
        std::vector<char> decompressed(chunkSize);
        std::vector<char> input(chunkSize+400);
        //read how many chunks are there
        uint32_t nChunks;
        str.read((char*)&nChunks,sizeof(nChunks));
        for(std::size_t i=0;i<nChunks;i++){
            str.read(&input[0],9);
            int c=qlz_size_compressed(&input[0]);
            str.read(&input[9],c-9);
            size_t d=qlz_decompress(&input[0], &decompressed[0], &state_decompress);
            decompressed_stream.write(&decompressed[0],d);
        }
        _used_str=&decompressed_stream;
    }
    else{
        _used_str=&str;
    }

    _used_str->read((char*)&m_k,sizeof(m_k));
    _used_str->read((char*)&m_L,sizeof(m_L));
    _used_str->read((char*)&m_scoring,sizeof(m_scoring));
    _used_str->read((char*)&m_weighting,sizeof(m_weighting));

    createScoringObject();
    m_nodes.resize(nnodes );
    m_nodes[0].id = 0;



    for(size_t i = 1; i < m_nodes.size(); ++i)
    {
        NodeId nid;
        _used_str->read((char*)&nid,sizeof(NodeId));
        Node& child = m_nodes[nid];
        child.id=nid;
        _used_str->read((char*)&child.parent,sizeof(child.parent));
        _used_str->read((char*)&child.weight,sizeof(child.weight));
        DescManip::fromStream(child.descriptor,*_used_str);
        m_nodes[child.parent].children.push_back(child.id);
     }
     //    // words
    uint32_t m_words_size;
    _used_str->read((char*)&m_words_size,sizeof(m_words_size));
    m_words.resize(m_words_size);
    for(unsigned int i = 0; i < m_words.size(); ++i)
    {
        WordId wid;NodeId nid;
        _used_str->read((char*)&wid,sizeof(wid));
        _used_str->read((char*)&nid,sizeof(nid));
        m_nodes[nid].word_id = wid;
        m_words[wid] = &m_nodes[nid];
    }
}
// --------------------------------------------------------------------------



void Vocabulary::load(const cv::FileStorage &fs,
  const std::string &name)
{
  m_words.clear();
  m_nodes.clear();

  cv::FileNode fvoc = fs[name];

  m_k = (int)fvoc["k"];
  m_L = (int)fvoc["L"];
  m_scoring = (ScoringType)((int)fvoc["scoringType"]);
  m_weighting = (WeightingType)((int)fvoc["weightingType"]);

  createScoringObject();

  // nodes
  cv::FileNode fn = fvoc["nodes"];

  m_nodes.resize(fn.size() + 1); // +1 to include root
  m_nodes[0].id = 0;

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId nid = (int)fn[i]["nodeId"];
    NodeId pid = (int)fn[i]["parentId"];
    WordValue weight = (WordValue)fn[i]["weight"];
    std::string d = (std::string)fn[i]["descriptor"];

    m_nodes[nid].id = nid;
    m_nodes[nid].parent = pid;
    m_nodes[nid].weight = weight;
    m_nodes[pid].children.push_back(nid);

    DescManip::fromString(m_nodes[nid].descriptor, d);
  }

  // words
  fn = fvoc["words"];

  m_words.resize(fn.size());

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId wid = (int)fn[i]["wordId"];
    NodeId nid = (int)fn[i]["nodeId"];

    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];
  }
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */

std::ostream& operator<<(std::ostream &os,
  const Vocabulary &voc)
{
  os << "Vocabulary: k = " << voc.getBranchingFactor()
    << ", L = " << voc.getDepthLevels()
    << ", Weighting = ";

  switch(voc.getWeightingType())
  {
    case TF_IDF: os << "tf-idf"; break;
    case TF: os << "tf"; break;
    case IDF: os << "idf"; break;
    case BINARY: os << "binary"; break;
  }

  os << ", Scoring = ";
  switch(voc.getScoringType())
  {
    case L1_NORM: os << "L1-norm"; break;
    case L2_NORM: os << "L2-norm"; break;
    case CHI_SQUARE: os << "Chi square distance"; break;
    case KL: os << "KL-divergence"; break;
    case BHATTACHARYYA: os << "Bhattacharyya coefficient"; break;
    case DOT_PRODUCT: os << "Dot product"; break;
  }

  os << ", Number of words = " << voc.size();

  return os;
}
/**
 * @brief Vocabulary::clear
 */
void Vocabulary::clear(){
    delete m_scoring_object;
    m_scoring_object=0;
    m_nodes.clear();
    m_words.clear();

}
int Vocabulary::getDescritorSize()const
{
    if (m_words.size()==0)return -1;
    else return m_words[0]->descriptor.cols;
}
int Vocabulary::getDescritorType()const{

    if (m_words.size()==0)return -1;
    else return m_words[0]->descriptor.type();
}


}
