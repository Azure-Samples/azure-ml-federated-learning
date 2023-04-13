#include <apsi/sender.h>
#include <apsi/receiver.h>
#include <apsi/util/cuckoo_filter.h>
#include <apsi/network/stream_channel.h>
#include <apsi/network/sender_operation.h>
#include <apsi/oprf/oprf_sender.h>
#include <apsi/oprf/oprf_receiver.h>
#include <apsi/log.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <future>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>

using namespace std;
using namespace apsi;
using namespace apsi::sender;
using namespace apsi::sender::util;
using namespace apsi::receiver;
using namespace apsi::oprf;
using namespace apsi::network;

namespace py = pybind11;

// Convert the items to the internal representation
vector<Item> to_items(const vector<string>& items)
{
    vector<Item> psi_items;
    psi_items.resize(items.size());
    std::for_each(begin(items), end(items), [&psi_items](string v) {
        psi_items.emplace_back(Item(v));
        });
    return psi_items;
}

class PSISender {
private:
    OPRFKey m_oprf_key;
    unique_ptr<CuckooFilter> m_filter;

public:
    PSISender(const vector<string>& items)
    {
        // Convert the items to the internal representation
        vector<Item> psi_items = to_items(items);

        // The sender first processes its items with the OPRF
        vector<HashedItem> hashed_items = OPRFSender::ComputeHashes(psi_items, this->m_oprf_key);

        // The sender will insert its items in the Cuckoo filter
        // The filter needs to be parameterized correctly to avoid false positives
        this->m_filter = make_unique<CuckooFilter>(items.size(), 60);
        for (auto& hashed_item : hashed_items) {
            this->m_filter->add(hashed_item.get_as<uint64_t>());
        }
        cout << "Sender finished inserting " << this->m_filter->get_num_items() << " items in the filter";
    }

    string get_serialized_filter()
    {
        stringstream exchange_stream;
        this->m_filter->save(exchange_stream);
        return exchange_stream.str();
    }

    string create_request_response(string request)
    {
        // Create channel for stroring data
        stringstream request_stream_in, request_stream_out;
        StreamChannel channel(request_stream_in, request_stream_out);

        // Receive and process the OPRF request
        stringstream input_stream;
        input_stream << request;

        auto sender_op = make_unique<SenderOperationOPRF>(SenderOperationOPRF());
        sender_op->load(input_stream);
        auto oprf_request = OPRFRequest(move(sender_op));
        // OPRFRequest oprf_request = to_oprf_request(move(request));

        Sender::RunOPRF(oprf_request, this->m_oprf_key, channel);

        stringstream output_stream;
        output_stream << request_stream_out.rdbuf();
        return output_stream.str();
    }

};

class PSIReceiver
{
private:
    OPRFReceiver m_oprf_receiver;
public:
    PSIReceiver(const vector<string>& items) : m_oprf_receiver{ Receiver::CreateOPRFReceiver(to_items(items)) } {}

    string create_request()
    {
		// Create the OPRF request
		auto oprf_request = Receiver::CreateOPRFRequest(m_oprf_receiver);
		// Send the OPRF request
		stringstream output_stream;
		oprf_request->save(output_stream);
		return output_stream.str();
	}

    vector<string> find_overlap(const string& cuckoo_filter, const string& response)
    {
        // Receive and process the OPRF request
        stringstream cuckoo_filter_stream;
        cuckoo_filter_stream << cuckoo_filter;

        size_t bytes_read;
        unique_ptr<CuckooFilter> filter = make_unique<CuckooFilter>(CuckooFilter::Load(cuckoo_filter_stream, bytes_read));

        // Receive and process the response
        stringstream request_stream_in, request_stream_out;
        StreamChannel channel(request_stream_in, request_stream_out);
        request_stream_in << response;

        OPRFResponse oprf_response = to_oprf_response(move(channel.receive_response()));
        auto receiver_oprf_items = Receiver::ExtractHashes(oprf_response, this->m_oprf_receiver).first;

        // Now all we need to do is check whether they appear in the filter
        size_t match_counter = 0;
        vector<string> items_intersection;
        for (auto& receiver_oprf_item : receiver_oprf_items) {
            if (filter->contains(receiver_oprf_item.get_as<uint64_t>())) {
                match_counter++;
                items_intersection.push_back(receiver_oprf_item.to_string());
            }
        }
        cout << "Receiver found " << match_counter << " matches";

        return items_intersection;
    }
};

PYBIND11_MODULE(psi, m) {
    m.doc() = "Minimal Symmetrical PSI library";

    py::class_<PSISender>(m, "PSISender")
        .def(py::init<const vector<string>&>())
        .def("get_serialized_filter", &PSISender::get_serialized_filter)
        .def("create_request_response", &PSISender::create_request_response);


    py::class_<PSIReceiver>(m, "PSIReceiver")
        .def(py::init<const vector<string>&>())
        .def("create_request", &PSIReceiver::create_request)
        .def("find_overlap", &PSIReceiver::find_overlap);
}
