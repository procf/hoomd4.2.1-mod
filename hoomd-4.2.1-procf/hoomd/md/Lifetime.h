//~ ########## Created by the Rheoinformatic research group ##########
//~ HOOMD-blue:
// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
//~
//~ This file:
///~ Written by Mohammad (Nabi) Nabizadeh and Dr. Deepak Mangal and Rob Campbell
///~ Documentation by Rob Campbell (2022)
///~ Extended to record per-particle tags and typeIDs, and to support either
///~ solvent-first or colloid-first tag orderings via a runtime colloid_offset.

#ifndef __LIFETIME_H__
#define __LIFETIME_H__

#include "hoomd/GlobalArray.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/Compute.h"

#include "hoomd/HOOMDMPI.h"

#include <memory>

#include <iostream>
#include <vector>
//~ #include <bits/stdc++.h> //~ comment out, not recommended
#include <fstream>
using namespace std;

namespace hoomd
    {
namespace md
    {
class Lifetime : public Compute
/* Records the formation and breaking of any colloid-colloid pair-particle bond during a simulation

	This is used to record the lifetime of each interparticle bond that is formed during a simulation

 */
    {
    public:
    // Constructor
    Lifetime(shared_ptr<SystemDefinition> sysdef):Compute(sysdef)
        {
        ArrayHandle<Scalar4> h_pos2(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
	// start the counter at zero
        num_colloid = 0;
	// initialize colloid_offset to a sentinel larger than any real tag
        colloid_offset = m_pdata->getNGlobal();
	// count the number of colloid particles AND find the smallest colloid tag
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            unsigned int typei_temp = __scalar_as_int(h_pos2.data[i].w);
            if (typei_temp != 0)
		{
                num_colloid += 1;
                if (h_tag.data[i] < colloid_offset)
                    {
                    colloid_offset = h_tag.data[i];
                    }
		}
            }

        // Determine the file names dynamically
        stringstream bondBrokeFileName;
        bondBrokeFileName << "BondBrokeHistory";
        //string fileNameBroke;

        int file_counter_broke = 0;
        while (true) {
            fileNameBroke = bondBrokeFileName.str();
            if (file_counter_broke > 0) {
                stringstream counter_broke;
                counter_broke << "_" << file_counter_broke;
                fileNameBroke += counter_broke.str();
            }
            fileNameBroke += ".csv";

            // Check if the file exists
            ifstream file(fileNameBroke.c_str());
            if (!file) {
                break; // Unique file name found, exit the loop
            }

            file_counter_broke++;
        }

        stringstream bondFormedFileName;
        bondFormedFileName << "BondFormedHistory";
        //string fileNameFormed;

        int file_counter_formed = 0;
        while (true) {
            fileNameFormed = bondFormedFileName.str();
            if (file_counter_formed > 0) {
                stringstream counter_formed;
                counter_formed << "_" << file_counter_formed;
                fileNameFormed += counter_formed.str();
            }
            fileNameFormed += ".csv";

            // Check if the file exists
            ifstream file(fileNameFormed.c_str());
            if (!file) {
                break; // Unique file name found, exit the loop
            }

            file_counter_formed++;
        }
    

    #ifdef ENABLE_MPI
        if(m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &num_colloid,
                          1,
                          MPI_INT,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            // global min over all ranks; ranks with no local colloids keep the
            // sentinel (getNGlobal()) and lose the MIN, so this is correct.
            MPI_Allreduce(MPI_IN_PLACE,
                          &colloid_offset,
                          1,
                          MPI_UNSIGNED,
                          MPI_MIN,
                          m_exec_conf->getMPICommunicator());
            }
    #endif

	// calculate the number of solvent particles
        num_solvent = m_pdata->getNGlobal() - num_colloid;

        // find the number of available j colloids to pair with i
        num_slot = num_colloid*(num_colloid-1)/2;

	// per-colloid type lookup, indexed by (tag - colloid_offset)
	// populated lazily from PotentialPairDPDThermo.h whenever a close pair is seen
        colloid_type.assign(num_colloid, 0);

	// get MPI rank/elements/index
        num_rank = m_exec_conf->getNRanks();
        my_rank = m_exec_conf->getRank();
        num_ele_per_process = num_slot / num_rank;
        index_start = my_rank * num_ele_per_process;
        index_end = (my_rank+1) * num_ele_per_process;
        if(my_rank == (num_rank-1))
	    {
            index_end = num_slot;
	    }
        track_size = index_end - index_start;
        Bond_track.resize(track_size);

	// create output files for recording bond data
        if(my_rank == 0)
            {
            BondBrokeFile.open(fileNameBroke.c_str(), ios::out | ios::app); //"BondBrokeHistory.csv",ios::out | ios::trunc);
            BondBrokeFile << "PairID,tagi,tagj,typei,typej,bondtime,broken at" << "\n";
            BondBrokeFile.close();
            BondFormedFile.open(fileNameFormed.c_str(), ios::out | ios::app); //"BondFormedHistory.csv",ios::out | ios::trunc);
            BondFormedFile << "PairID,tagi,tagj,typei,typej,formed at" << "\n";
            BondFormedFile.close();
            }
        }

    /// Destructor
    ~Lifetime(){}

    //update the bond lifetime at each timestep
    virtual void updatebondtime(uint64_t timestep)
        {
        vector<map<unsigned int, unsigned char> > Bond_check_temp1(num_rank);
        all_gather_v(Bond_check, Bond_check_temp1, m_exec_conf->getMPICommunicator());
        map<unsigned int, unsigned char> Bond_check_temp2;
        for (unsigned int i=0; i < num_rank; i++)
            {
            for(auto x : Bond_check_temp1[i])
                {
                Bond_check_temp2[x.first]=x.second;
                }
            }
        Bond_check.clear();
        Bond_check_temp1.clear();
        for (unsigned int i = 0; i < track_size; i++)
            {
            unsigned int j = i + index_start;
            if(Bond_check_temp2.count(j))
                {
                if(Bond_check_temp2[j] == 2)
                    {
                    if(Bond_track[i] == 0)
                        {
                        Bond_track[i] = 1;
                        Bond_formed.push_back({j,(unsigned int)timestep});
                        }
                    else
			{
                        Bond_track[i] += 1;
			}
                    }
                else
                    {
                    if(Bond_track[i] != 0)
			{
                        Bond_track[i] += 1;
			}
                    }
                }
            else
                {
                if(Bond_track[i] != 0)
                    {
                    Bond_broke.push_back({j,Bond_track[i],(unsigned int)timestep});
                    Bond_track[i] = 0;
                    }
                }
            }
        Bond_check_temp2.clear();
        }


    // record the lifetime at which a bond breaks/forms
    virtual void writeBondtime()
        {
    #ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            // Make sure rank 0 has the full type table before writing.
            // Each rank's entry is either 0 (never seen) or the true type;
            // MAX correctly picks the real type wherever it was set.
            MPI_Allreduce(MPI_IN_PLACE,
                          colloid_type.data(),
                          (int)num_colloid,
                          MPI_UNSIGNED_CHAR,
                          MPI_MAX,
                          m_exec_conf->getMPICommunicator());

            vector<vector<uint3> > Bond_broke_temp;
            gather_v(Bond_broke, Bond_broke_temp, 0, m_exec_conf->getMPICommunicator());
            Bond_broke.clear();

            vector<vector<uint2> > Bond_formed_temp;
            gather_v(Bond_formed, Bond_formed_temp, 0, m_exec_conf->getMPICommunicator());
            Bond_formed.clear();
            if(my_rank == 0)
                {   
    	        BondBrokeFile.open(fileNameBroke.c_str(), ios::out | ios::app); //"BondBrokeHistory.csv",ios::app);
                for (unsigned int i=0; i < num_rank; i++)
                    {
    		    for (unsigned int j=0; j < Bond_broke_temp[i].size(); j++)
    		        {
                        unsigned int bidx = Bond_broke_temp[i][j].x;
                        unsigned int v1, v2;
                        decode_bond(bidx, v1, v2);
                        unsigned int tagi_out = v1 + colloid_offset;
                        unsigned int tagj_out = v2 + colloid_offset;
                        unsigned int ti = (unsigned int)colloid_type[v1];
                        unsigned int tj = (unsigned int)colloid_type[v2];
    		        BondBrokeFile << bidx       << ","
                                      << tagi_out  << "," << tagj_out << ","
                                      << ti        << "," << tj       << ","
                                      << Bond_broke_temp[i][j].y << ","
                                      << Bond_broke_temp[i][j].z << "\n";
    		        }
                    }
                Bond_broke_temp.clear();
    	        BondBrokeFile.close();

                BondFormedFile.open(fileNameFormed.c_str(), ios::out | ios::app); //"BondFormedHistory.csv",ios::app);
                for (unsigned int i=0; i < num_rank; i++)
                    {
                    for (unsigned int j=0; j < Bond_formed_temp[i].size(); j++)
                        {
                        unsigned int bidx = Bond_formed_temp[i][j].x;
                        unsigned int v1, v2;
                        decode_bond(bidx, v1, v2);
                        unsigned int tagi_out = v1 + colloid_offset;
                        unsigned int tagj_out = v2 + colloid_offset;
                        unsigned int ti = (unsigned int)colloid_type[v1];
                        unsigned int tj = (unsigned int)colloid_type[v2];
                        BondFormedFile << bidx       << ","
                                       << tagi_out  << "," << tagj_out << ","
                                       << ti        << "," << tj       << ","
                                       << Bond_formed_temp[i][j].y << "\n";
                        }
                    }
                Bond_formed_temp.clear();
                BondFormedFile.close();
    	        }
            }
        else
    	    {
            BondBrokeFile.open(fileNameBroke.c_str(), ios::out | ios::app); //"BondBrokeHistory.csv",ios::app);
            for (unsigned int i=0; i < Bond_broke.size(); i++)
                {
                unsigned int bidx = Bond_broke[i].x;
                unsigned int v1, v2;
                decode_bond(bidx, v1, v2);
                BondBrokeFile << bidx                            << ","
                              << (v1 + colloid_offset)           << ","
                              << (v2 + colloid_offset)           << ","
                              << (unsigned int)colloid_type[v1]  << ","
                              << (unsigned int)colloid_type[v2]  << ","
                              << Bond_broke[i].y                 << ","
                              << Bond_broke[i].z                 << "\n";
                }
    	    Bond_broke.clear();
    	    BondBrokeFile.close();
    
            BondFormedFile.open(fileNameFormed.c_str(), ios::out | ios::app); //"BondFormedHistory.csv",ios::app);
            for (unsigned int i=0; i < Bond_formed.size(); i++)
                {
                unsigned int bidx = Bond_formed[i].x;
                unsigned int v1, v2;
                decode_bond(bidx, v1, v2);
                BondFormedFile << bidx                            << ","
                               << (v1 + colloid_offset)           << ","
                               << (v2 + colloid_offset)           << ","
                               << (unsigned int)colloid_type[v1]  << ","
                               << (unsigned int)colloid_type[v2]  << ","
                               << Bond_formed[i].y                << "\n";
                }
            Bond_formed.clear();
            BondFormedFile.close();
    	    }
    #endif
        }
    unsigned int num_colloid;         //!< the number of colloid particles
    unsigned int num_solvent;         //!< the number of solvent particles
    unsigned int num_slot;            //!< number of j colloids able to pair with a colloid particle i
    unsigned int num_rank;            //!< number of MPI ranks
    unsigned int my_rank;             //!< current rank
    unsigned int num_ele_per_process; //!< number of elements per MPI process
    unsigned int index_start;         //!< elements in this process (start)
    unsigned int index_end;           //!< elements in this process (end)
    unsigned int track_size;          //!< size of a process
    unsigned int colloid_offset;      //!< smallest tag belonging to a colloid; subtract from a tag to get a 0..num_colloid-1 index
    map<unsigned int, unsigned char> Bond_check;
    vector<unsigned int> Bond_track;
    vector<uint3> Bond_broke;
    vector<uint2> Bond_formed;
    vector<unsigned char> colloid_type; //!< type of each colloid, indexed by (tag - colloid_offset); populated lazily from PotentialPairDPDThermo
    protected:
    ofstream BondFormedFile;
    ofstream BondBrokeFile;
    std::string fileNameBroke;        //!< File name for bond broke history
    std::string fileNameFormed;       //!< File name for bond formed history

    // Recover (var1, var2) from a bond_index produced by:
    //     bond_index = num_colloid*var1 - var1*(var1+1)/2 + var2 - var1 - 1
    // with 0 <= var1 < var2 < num_colloid.
    inline void decode_bond(unsigned int bond_index,
                            unsigned int& var1,
                            unsigned int& var2) const
        {
        var1 = 0;
        unsigned int row_start = 0;
        while (var1 + 1 < num_colloid)
            {
            unsigned int row_count = num_colloid - 1 - var1;
            if (bond_index < row_start + row_count) break;
            row_start += row_count;
            var1++;
            }
        var2 = bond_index - row_start + var1 + 1;
        }
    };
 
    } // end namespace md
    } // end namespace hoomd

#endif // __LIFETIME_H__
